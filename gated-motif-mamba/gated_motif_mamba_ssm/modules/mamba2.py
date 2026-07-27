# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from gated_motif_mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from gated_motif_mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from gated_motif_mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref

from gated_motif_mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan, mamba_chunk_scan_combined
from gated_motif_mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        input_gate_on_state=False,
        input_gate_mode="exp_decay",
        input_gate_gain=0.5,
        input_gate_chunk_reduce="last",
        input_gate_chunk_temp=4.0,
        state_pq_rank=0,
        state_pq_alpha=0.0,
        state_pq_apply_to="running",
        state_pq_headwise=False,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        if process_group is not None:
            raise NotImplementedError("Trimmed build only supports process_group=None")
        self.world_size = 1
        self.local_rank = 0
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.state_pq_dim = self.headdim * self.d_state
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.input_gate_on_state = input_gate_on_state
        self.input_gate_mode = input_gate_mode
        self.input_gate_gain = float(input_gate_gain)
        self.input_gate_chunk_reduce = input_gate_chunk_reduce
        self.input_gate_chunk_temp = float(input_gate_chunk_temp)
        self.state_pq_rank = int(state_pq_rank)
        self.state_pq_alpha = float(state_pq_alpha)
        self.state_pq_apply_to = state_pq_apply_to
        self.state_pq_headwise = state_pq_headwise
        self.layer_idx = layer_idx
        if self.input_gate_mode not in {"exp_decay", "symmetric_exp"}:
            raise ValueError("input_gate_mode must be one of {'exp_decay', 'symmetric_exp'}")
        if self.input_gate_chunk_reduce not in {"last", "importance"}:
            raise ValueError("input_gate_chunk_reduce must be one of {'last', 'importance'}")
        if self.state_pq_apply_to not in {"new", "running", "both"}:
            raise ValueError("state_pq_apply_to must be one of {'new', 'running', 'both'}")
        self.state_pq_enabled = self.state_pq_rank > 0 and self.state_pq_alpha != 0.0

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()
        if self.input_gate_on_state:
            self.input_gate_proj = nn.Linear(self.headdim, 1, bias=True, **factory_kwargs)
        if self.state_pq_rank > 0:
            if self.state_pq_headwise:
                self.state_pq_P = nn.Parameter(
                    torch.empty(self.nheads, self.state_pq_dim, self.state_pq_rank, **factory_kwargs)
                )
                self.state_pq_Q = nn.Parameter(
                    torch.zeros(self.nheads, self.state_pq_rank, self.state_pq_dim, **factory_kwargs)
                )
            else:
                self.state_pq_P = nn.Parameter(torch.empty(self.state_pq_dim, self.state_pq_rank, **factory_kwargs))
                self.state_pq_Q = nn.Parameter(torch.zeros(self.state_pq_rank, self.state_pq_dim, **factory_kwargs))
            nn.init.normal_(self.state_pq_P, mean=0.0, std=1.0 / math.sqrt(max(self.state_pq_rank, 1)))
        else:
            self.state_pq_P = None
            self.state_pq_Q = None

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(
        self,
        u,
        seqlen=None,
        seq_idx=None,
        cu_seqlens=None,
        inference_params=None,
        return_ssm_state=False,
        return_chunk_state=False,
    ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        if return_ssm_state and return_chunk_state:
            raise ValueError("return_ssm_state and return_chunk_state cannot both be True")
        if return_ssm_state:
            return self._forward_with_state_tracking(
                u,
                seqlen=seqlen,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                inference_params=inference_params,
            )
        if return_chunk_state and (seq_idx is not None or cu_seqlens is not None or inference_params is not None):
            raise NotImplementedError("return_chunk_state does not support seq_idx/cu_seqlens/inference_params")
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            if self.state_pq_enabled and self.state_pq_apply_to in {"running", "both"}:
                raise NotImplementedError("inference_params/step decoding is not supported when state_pq is enabled")
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        use_chunk_scan_path = self.input_gate_on_state or self.state_pq_enabled
        if return_chunk_state and not use_chunk_scan_path:
            raise NotImplementedError("return_chunk_state requires the chunk scan path")
        if self.use_mem_eff_path and inference_params is None and not use_chunk_scan_path:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
            B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
            C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
            z_ssm = rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None
            if use_chunk_scan_path:
                assert seq_idx is None, "chunk state passing with input gate or state_pq does not support seq_idx yet"
                assert cu_seqlens is None, "chunk state passing with input gate or state_pq does not support cu_seqlens yet"
                input_gate = self._chunk_input_gate(x) if self.input_gate_on_state else None
                if return_chunk_state:
                    y, chunk_states, chunk_final_state = mamba_chunk_scan(
                        x,
                        dt,
                        A,
                        B,
                        C,
                        chunk_size=self.chunk_size,
                        D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                        z=z_ssm,
                        dt_bias=self.dt_bias,
                        dt_softplus=True,
                        initial_states=ssm_state if ssm_state is not None else None,
                        input_gate=input_gate,
                        pq_P=self.state_pq_P,
                        pq_Q=self.state_pq_Q,
                        pq_alpha=self.state_pq_alpha,
                        pq_apply_to=self.state_pq_apply_to,
                        return_chunk_states=True,
                        return_final_states=True,
                    )
                else:
                    y = mamba_chunk_scan(
                        x,
                        dt,
                        A,
                        B,
                        C,
                        chunk_size=self.chunk_size,
                        D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                        z=z_ssm,
                        dt_bias=self.dt_bias,
                        dt_softplus=True,
                        initial_states=ssm_state if ssm_state is not None else None,
                        input_gate=input_gate,
                        pq_P=self.state_pq_P,
                        pq_Q=self.state_pq_Q,
                        pq_alpha=self.state_pq_alpha,
                        pq_apply_to=self.state_pq_apply_to,
                        return_final_states=ssm_state is not None,
                    )
            else:
                y = mamba_chunk_scan_combined(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=z_ssm,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    cu_seqlens=cu_seqlens,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                    return_varlen_states=cu_seqlens is not None and inference_params is not None,
                )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self._apply_rmsnorm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        if return_chunk_state:
            return out, chunk_states, chunk_final_state
        return out

    def _forward_with_state_tracking(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        if seqlen is not None:
            raise NotImplementedError("return_ssm_state only supports batched (B, L, D) inputs")
        if seq_idx is not None or cu_seqlens is not None or inference_params is not None:
            raise NotImplementedError("return_ssm_state does not support seq_idx/cu_seqlens/inference_params")
        if u.ndim != 3:
            raise ValueError("Expected u to have shape (batch, seqlen, hidden_dim)")

        batch, seqlen, _ = u.shape
        conv_state, ssm_state = self.allocate_inference_cache(batch, seqlen, dtype=u.dtype)
        outputs = []
        ssm_states = []
        for t in range(seqlen):
            out_t, conv_state, ssm_state = self.step(u[:, t : t + 1], conv_state, ssm_state)
            outputs.append(out_t)
            ssm_states.append(ssm_state.clone())
        out = torch.cat(outputs, dim=1)
        ssm_states = torch.stack(ssm_states, dim=1)
        return out, ssm_states

    def _apply_rmsnorm(self, y, z):
        if y.is_cuda:
            return self.norm(y, z)
        return rms_norm_ref(
            y,
            self.norm.weight,
            self.norm.bias,
            z=z,
            eps=self.norm.eps,
            group_size=self.norm.group_size,
            norm_before_gate=self.norm.norm_before_gate,
        )

    def _apply_state_pq_to_ssm_state(self, state):
        if not self.state_pq_enabled:
            return state
        state_dtype = state.dtype
        state_flat = rearrange(state.to(torch.float32), "b h p n -> b h (p n)")
        if self.state_pq_P.dim() == 2:
            state_low = torch.einsum("bhd,dr->bhr", state_flat, self.state_pq_P.to(dtype=state_flat.dtype))
            state_mix = torch.einsum("bhr,rd->bhd", state_low, self.state_pq_Q.to(dtype=state_flat.dtype))
        else:
            state_low = torch.einsum("bhd,hdr->bhr", state_flat, self.state_pq_P.to(dtype=state_flat.dtype))
            state_mix = torch.einsum("bhr,hrd->bhd", state_low, self.state_pq_Q.to(dtype=state_flat.dtype))
        state_flat = state_flat + float(self.state_pq_alpha) * torch.tanh(state_mix)
        return rearrange(state_flat.to(dtype=state_dtype), "b h (p n) -> b h p n", p=self.headdim, n=self.d_state)

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None or not xBC.is_cuda:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None or self.input_gate_on_state or self.state_pq_enabled:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if self.input_gate_on_state:
                dA = dA * self._compute_input_gate(x)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            prev_state = ssm_state.to(torch.float32)
            new_state = dBx.to(torch.float32)
            if self.state_pq_enabled:
                if self.state_pq_apply_to in {"running", "both"}:
                    prev_state = self._apply_state_pq_to_ssm_state(prev_state).to(torch.float32)
                if self.state_pq_apply_to in {"new", "both"}:
                    new_state = self._apply_state_pq_to_ssm_state(new_state).to(torch.float32)
            updated_state = prev_state * rearrange(dA.to(torch.float32), "b h -> b h 1 1") + new_state
            ssm_state.copy_(updated_state.to(dtype=ssm_state.dtype))
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self._apply_rmsnorm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def _compute_input_gate_logits(self, x):
        return self.input_gate_proj(x).squeeze(-1).float()

    def _transform_input_gate_logits(self, gate_logits):
        if self.input_gate_mode == "exp_decay":
            return torch.exp(-torch.exp(gate_logits))
        # Symmetric multiplicative gate centered at 1.0. Positive logits amplify,
        # negative logits attenuate, while exp(.) keeps the scale strictly positive.
        return torch.exp(self.input_gate_gain * torch.tanh(gate_logits))

    def _compute_input_gate(self, x):
        gate_logits = self._compute_input_gate_logits(x)
        return self._transform_input_gate_logits(gate_logits)

    def _chunk_input_gate(self, x):
        gate_logits = self._compute_input_gate_logits(x)
        batch, seqlen, _, _ = x.shape
        pad_len = (self.chunk_size - seqlen % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            gate_logits = F.pad(gate_logits, (0, 0, 0, pad_len), value=0.0)
        gate_logits = rearrange(gate_logits, "b (c l) h -> b h c l", l=self.chunk_size)
        if self.input_gate_chunk_reduce == "importance":
            # Let salient positions dominate the chunk gate instead of always using
            # the last token. The sign of the pooled logits decides amplify vs. attenuate.
            scores = self.input_gate_chunk_temp * gate_logits.abs()
            weights = torch.softmax(scores, dim=-1)
            chunk_logits = (weights * gate_logits).sum(dim=-1)
            return self._transform_input_gate_logits(chunk_logits)
        # Backward-compatible behavior: use the last valid token in each chunk as the
        # transition gate for the chunk-to-chunk state update.
        if pad_len == 0:
            return self._transform_input_gate_logits(gate_logits[..., -1])
        valid_lengths = torch.full(
            (gate_logits.shape[2],), self.chunk_size, device=gate_logits.device, dtype=torch.long
        )
        valid_lengths[-1] = self.chunk_size - pad_len
        gather_idx = (valid_lengths - 1).view(1, 1, -1, 1).expand(batch, self.nheads, -1, 1)
        chunk_logits = torch.gather(gate_logits, -1, gather_idx).squeeze(-1)
        return self._transform_input_gate_logits(chunk_logits)

    def get_state_pq_square(self):
        if self.state_pq_P is None or self.state_pq_Q is None:
            return None
        return torch.matmul(self.state_pq_P, self.state_pq_Q)

    def get_state_pq_motif_matrix(self):
        pq_square = self.get_state_pq_square()
        if pq_square is None:
            return None
        if pq_square.dim() == 2:
            pq_square = pq_square.view(self.headdim, self.d_state, self.headdim, self.d_state)
            return pq_square.mean(dim=(0, 2))
        if pq_square.dim() == 3:
            pq_square = pq_square.view(self.nheads, self.headdim, self.d_state, self.headdim, self.d_state)
            return pq_square.mean(dim=(1, 3))
        raise ValueError(f"Unsupported state PQ shape: {tuple(pq_square.shape)}")

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
