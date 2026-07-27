# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from gated_motif_mamba_ssm.utils.determinism import autotune_configs


@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ]),
    key=['dim'],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr, input_gate_ptr, initstates_ptr, seq_idx_ptr,
    # Matrix dimensions
    dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_input_gate_batch, stride_input_gate_chunk, stride_input_gate_head,
    stride_initstates_batch, stride_initstates_head, stride_initstates_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INPUT_GATE: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    if HAS_INPUT_GATE:
        input_gate_ptr += pid_b * stride_input_gate_batch + pid_h * stride_input_gate_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    seq_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_INPUT_GATE:
            input_gate = tl.load(input_gate_ptr).to(tl.float32)
            scale *= input_gate
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        if HAS_INPUT_GATE:
            input_gate_ptr += stride_input_gate_chunk
        out_ptrs += stride_out_chunk


@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ]),
    key=['dim'],
)
@triton.jit
def _state_passing_bwd_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, dA_cs_ptr, input_gate_ptr, dfinal_states_ptr, seq_idx_ptr,
    dstates_ptr, ddA_cs_ptr, dinput_gate_ptr, dinitstates_ptr, states_converted_ptr,
    # Matrix dimensions
    dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_input_gate_batch, stride_input_gate_chunk, stride_input_gate_head,
    stride_dfinal_states_batch, stride_dfinal_states_head, stride_dfinal_states_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_dim,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head,
    stride_dinput_gate_batch, stride_dinput_gate_chunk, stride_dinput_gate_head,
    stride_dinitstates_batch, stride_dinitstates_head, stride_dinitstates_dim,
    # Meta-parameters
    HAS_INPUT_GATE: tl.constexpr,
    CONVERT_STATES: tl.constexpr,
    HAS_DFINAL_STATES: tl.constexpr,
    HAS_DINITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dstates_ptr += pid_b * stride_dstates_batch + pid_h * stride_dstates_head + (nchunks - 1) * stride_dstates_chunk
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head + (nchunks - 1) * stride_dA_cs_chunk
    ddA_cs_ptr += pid_b * stride_ddA_cs_batch + pid_h * stride_ddA_cs_head + (nchunks - 1) * stride_ddA_cs_chunk + pid_m
    if HAS_INPUT_GATE:
        input_gate_ptr += pid_b * stride_input_gate_batch + pid_h * stride_input_gate_head + (nchunks - 1) * stride_input_gate_chunk
        dinput_gate_ptr += pid_b * stride_dinput_gate_batch + pid_h * stride_dinput_gate_head + (nchunks - 1) * stride_dinput_gate_chunk + pid_m
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + (nchunks - 1) * stride_out_chunk
    dout_ptr += pid_b * stride_dout_batch + pid_h * stride_dout_head + (nchunks - 1) * stride_dout_chunk
    if CONVERT_STATES:
        states_converted_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + (nchunks - 1) * stride_out_chunk
    if HAS_DFINAL_STATES:
        dfinal_states_ptr += pid_b * stride_dfinal_states_batch + pid_h * stride_dfinal_states_head
    if HAS_DINITSTATES:
        dinitstates_ptr += pid_b * stride_dinitstates_batch + pid_h * stride_dinitstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dstates_ptrs = dstates_ptr + offs_m * stride_dstates_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    dout_ptrs = dout_ptr + offs_m * stride_dout_dim
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_m * stride_out_dim

    if HAS_DFINAL_STATES:
        dstates = tl.load(dfinal_states_ptr + offs_m * stride_dfinal_states_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        dstates = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + (seqlen - 1) * stride_seq_idx_seqlen)
    dstates_ptrs -= stride_dstates_chunk
    for c in range(nchunks - 1):
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_INPUT_GATE:
            input_gate = tl.load(input_gate_ptr).to(tl.float32)
            scale *= input_gate
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (((nchunks - c - 1) * chunk_size - 1) * stride_seq_idx_seqlen))
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=offs_m < dim)
        dot = tl.sum(out * dstates)
        ddA = dot * scale
        tl.store(ddA_cs_ptr, ddA)
        if HAS_INPUT_GATE:
            dinput_gate = dot * tl.exp(dA_cs)
            if HAS_SEQ_IDX:
                dinput_gate = tl.where(seq_idx_new == seq_idx, dinput_gate, 0.0)
            tl.store(dinput_gate_ptr, dinput_gate)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
        dout_ptrs -= stride_dout_chunk
        dstates_ptrs -= stride_dstates_chunk
        dA_cs_ptr -= stride_dA_cs_chunk
        ddA_cs_ptr -= stride_ddA_cs_chunk
        if HAS_INPUT_GATE:
            input_gate_ptr -= stride_input_gate_chunk
            dinput_gate_ptr -= stride_dinput_gate_chunk
        out_ptrs -= stride_out_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk
    if CONVERT_STATES:
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(states_converted_ptrs, out, mask=offs_m < dim)
    if not HAS_DINITSTATES:
        tl.store(ddA_cs_ptr, 0.0)
        if HAS_INPUT_GATE:
            tl.store(dinput_gate_ptr, 0.0)
    else:
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_INPUT_GATE:
            input_gate = tl.load(input_gate_ptr).to(tl.float32)
            scale *= input_gate
        if HAS_SEQ_IDX:
            scale = tl.where(seq_idx == 0, scale, 0.0)
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dot = tl.sum(out * dstates)
        ddA = dot * scale
        tl.store(ddA_cs_ptr, ddA)
        if HAS_INPUT_GATE:
            dinput_gate = dot * tl.exp(dA_cs)
            if HAS_SEQ_IDX:
                dinput_gate = tl.where(seq_idx == 0, dinput_gate, 0.0)
            tl.store(dinput_gate_ptr, dinput_gate)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dinitstates_ptr + offs_m * stride_dinitstates_dim, dstates, mask=offs_m < dim)


def _state_passing_fwd(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None,
                       input_gate=None, out_dtype=None):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if input_gate is not None:
        assert input_gate.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.float32)
    if input_gate is not None:
        input_gate = input_gate.contiguous()
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states, out, final_states, dA_chunk_cumsum, input_gate if input_gate is not None else states,
            initial_states, seq_idx,
            dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_states.stride(0), final_states.stride(1), final_states.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((input_gate.stride(0), input_gate.stride(2), input_gate.stride(1))
              if input_gate is not None else (0, 0, 0)),
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
              if initial_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_INPUT_GATE=input_gate is not None,
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out, final_states


def _state_passing_bwd(
        states, dA_chunk_cumsum, dout, dfinal_states=None, seq_idx=None, has_initial_states=None,
        input_gate=None, dstates_dtype=None, states_dtype=None, chunk_size=None
):
    """
    states contains the initial_states at index 0. The final states are not included in states.
    """
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    assert dout.shape == (batch, nchunks, nheads, dim)
    if input_gate is not None:
        assert input_gate.shape == (batch, nheads, nchunks)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    dstates = torch.empty_like(dout, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
    if input_gate is not None:
        dinput_gate = torch.empty_like(dA_chunk_cumsum, dtype=torch.float32)
    else:
        dinput_gate = None
    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(states, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None
    if has_initial_states:
        dinitstates = torch.empty_like(dstates[:, 0])
    else:
        dinitstates = None
    if dfinal_states is not None:
        assert dfinal_states.shape == (batch, nheads, dim)
    BLOCK_SIZE_min = 64
    n_blocks = (dim + BLOCK_SIZE_min - 1) // BLOCK_SIZE_min
    ddA_chunk_cumsum = torch.empty(batch, nheads, nchunks, n_blocks,
                                    dtype=torch.float32, device=dA_chunk_cumsum.device)
    if input_gate is not None:
        dinput_gate = torch.empty(batch, nheads, nchunks, n_blocks, dtype=torch.float32, device=dA_chunk_cumsum.device)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(dout.device.index):
        _state_passing_bwd_kernel[grid](
            dout, states, dA_chunk_cumsum, input_gate if input_gate is not None else states, dfinal_states, seq_idx,
            dstates, ddA_chunk_cumsum, dinput_gate if input_gate is not None else states, dinitstates, states_converted,
            dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((input_gate.stride(0), input_gate.stride(2), input_gate.stride(1))
              if input_gate is not None else (0, 0, 0)),
            *((dfinal_states.stride(0), dfinal_states.stride(1), dfinal_states.stride(2))
                if dfinal_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3),
            ddA_chunk_cumsum.stride(0), ddA_chunk_cumsum.stride(2), ddA_chunk_cumsum.stride(1),
            *((dinput_gate.stride(0), dinput_gate.stride(2), dinput_gate.stride(1))
              if input_gate is not None else (0, 0, 0)),
            *((dinitstates.stride(0), dinitstates.stride(1), dinitstates.stride(2))
              if dinitstates is not None else (0, 0, 0)),
            HAS_INPUT_GATE=input_gate is not None,
            CONVERT_STATES=states_converted is not None,
            HAS_DFINAL_STATES=dfinal_states is not None,
            HAS_DINITSTATES=dinitstates is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    BLOCK_SIZE_actual = _state_passing_bwd_kernel.best_config.kwargs["BLOCK_SIZE"]
    n_valid_blocks = (dim + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    ddA_chunk_cumsum = ddA_chunk_cumsum[..., :n_valid_blocks].sum(dim=-1).to(dtype=dA_chunk_cumsum.dtype)
    if input_gate is not None:
        dinput_gate = dinput_gate[..., :n_valid_blocks].sum(dim=-1).to(dtype=dA_chunk_cumsum.dtype)
    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states
    if input_gate is None:
        return (dstates, ddA_chunk_cumsum, dinitstates) if states_dtype is None else (dstates, ddA_chunk_cumsum, dinitstates, states_converted)
    return (
        dstates,
        ddA_chunk_cumsum,
        dinitstates,
        dinput_gate,
    ) if states_dtype is None else (
        dstates,
        ddA_chunk_cumsum,
        dinitstates,
        dinput_gate,
        states_converted,
    )


class StatePassingFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, states, dA_chunk_cumsum, initial_states=None, input_gate=None):
        batch, nchunks, nheads, dim = states.shape
        assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
        if input_gate is not None:
            assert input_gate.shape == (batch, nheads, nchunks)
        if states.stride(-1) != 1:
            states = states.contiguous()
        out, final_states = _state_passing_fwd(states, dA_chunk_cumsum, initial_states, input_gate=input_gate)
        if input_gate is None:
            ctx.save_for_backward(out, dA_chunk_cumsum)
        else:
            ctx.save_for_backward(out, dA_chunk_cumsum, input_gate)
        ctx.has_initial_states = initial_states is not None
        ctx.has_input_gate = input_gate is not None
        return out, final_states

    @staticmethod
    def backward(ctx, dout, dfinal_states):
        if ctx.has_input_gate:
            out, dA_chunk_cumsum, input_gate = ctx.saved_tensors
        else:
            out, dA_chunk_cumsum = ctx.saved_tensors
            input_gate = None
        batch, nchunks, nheads, dim = out.shape
        assert dout.shape == (batch, nchunks, nheads, dim)
        assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
        assert dfinal_states.shape == (batch, nheads, dim)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if input_gate is None:
            dstates, ddA_chunk_cumsum, dinitstates = _state_passing_bwd(
                out, dA_chunk_cumsum, dout, dfinal_states=dfinal_states, has_initial_states=ctx.has_initial_states
            )
            return dstates, ddA_chunk_cumsum, dinitstates, None
        dstates, ddA_chunk_cumsum, dinitstates, dinput_gate = _state_passing_bwd(
            out, dA_chunk_cumsum, dout, dfinal_states=dfinal_states, has_initial_states=ctx.has_initial_states, input_gate=input_gate
        )
        return dstates, ddA_chunk_cumsum, dinitstates, dinput_gate


def state_passing(
    states,
    dA_chunk_cumsum,
    initial_states=None,
    input_gate=None,
    pq_dt=None,
    pq_P=None,
    pq_Q=None,
    pq_alpha=0.0,
    pq_apply_to="running",
):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    needs_grad = torch.is_grad_enabled() and any(
        tensor is not None and getattr(tensor, "requires_grad", False)
        for tensor in (states, initial_states, input_gate, pq_P, pq_Q)
    )
    has_pq = pq_P is not None or pq_Q is not None
    # Training with PQ applied to the per-chunk new state is numerically more
    # stable on the eager recurrence path. We keep the Triton recurrence for
    # inference / no-grad execution.
    if has_pq and pq_apply_to == "new" and needs_grad:
        return state_passing_gated(
            states,
            dA_chunk_cumsum,
            initial_states=initial_states,
            input_gate=input_gate,
            pq_dt=pq_dt,
            pq_P=pq_P,
            pq_Q=pq_Q,
            pq_alpha=pq_alpha,
            pq_apply_to="new",
        )
    if has_pq and pq_apply_to in {"new", "both"}:
        batch, nchunks, nheads, dim = states.shape
        flat_states = rearrange(states, "b c h d -> (b c) h d").contiguous()
        flat_states = flat_states + _compute_low_rank_pq_delta(flat_states, pq_P, pq_Q, pq_alpha)
        states = rearrange(flat_states, "(b c) h d -> b c h d", b=batch, c=nchunks).contiguous()
    # Keep input-gate-only execution on the Triton-backed path. PQ can also stay on
    # the Triton recurrence when it only modifies the per-chunk new state.
    if has_pq and pq_apply_to in {"running", "both"}:
        return state_passing_gated(
            states,
            dA_chunk_cumsum,
            initial_states=initial_states,
            input_gate=input_gate,
            pq_dt=pq_dt,
            pq_P=pq_P,
            pq_Q=pq_Q,
            pq_alpha=pq_alpha,
            pq_apply_to="running",
        )
    return StatePassingFn.apply(states, dA_chunk_cumsum, initial_states, input_gate)


def _compute_low_rank_pq_delta(x, pq_P, pq_Q, pq_alpha, pq_gate=None, pq_dt=None):
    if pq_P is None and pq_Q is None:
        return torch.zeros_like(x)
    if pq_P is None or pq_Q is None:
        raise ValueError("pq_P and pq_Q must be both set or both None")
    if pq_alpha == 0.0:
        return torch.zeros_like(x)

    if pq_P.dim() == 2:
        dim, rank = pq_P.shape
        assert pq_Q.shape == (rank, dim)
        assert x.shape[-1] == dim
        x_low = torch.einsum("bhd,dr->bhr", x, pq_P.to(dtype=x.dtype))
        x_mix = torch.einsum("bhr,rd->bhd", x_low, pq_Q.to(dtype=x.dtype))
    elif pq_P.dim() == 3:
        nheads, dim, rank = pq_P.shape
        assert pq_Q.shape == (nheads, rank, dim)
        assert x.shape[1] == nheads
        assert x.shape[-1] == dim
        x_low = torch.einsum("bhd,hdr->bhr", x, pq_P.to(dtype=x.dtype))
        x_mix = torch.einsum("bhr,hrd->bhd", x_low, pq_Q.to(dtype=x.dtype))
    else:
        raise ValueError(f"Unsupported pq_P rank: expected 2D or 3D, got shape {tuple(pq_P.shape)}")

    delta = float(pq_alpha) * x_mix
    if pq_gate is not None:
        delta = pq_gate[:, :, None].to(dtype=delta.dtype) * delta
    if pq_dt is not None:
        delta = pq_dt[:, :, None].to(dtype=delta.dtype) * delta
    return delta


def state_passing_gated(
    states,
    dA_chunk_cumsum,
    initial_states=None,
    input_gate=None,
    pq_dt=None,
    seq_idx=None,
    chunk_size=None,
    pq_P=None,
    pq_Q=None,
    pq_alpha=0.0,
    pq_apply_to="running",
):
    """
    Pure PyTorch recurrence for input-gated state passing.

    This path is slower than the Triton kernel but keeps gradients for `input_gate`
    in eager autograd, which makes it a practical path for experimenting with new
    recurrent updates.
    """
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if pq_apply_to not in {"new", "running", "both"}:
        raise ValueError("pq_apply_to must be one of {'new', 'running', 'both'}")
    if initial_states is None:
        running = torch.zeros(batch, nheads, dim, device=states.device, dtype=torch.float32)
    else:
        assert initial_states.shape == (batch, nheads, dim)
        running = initial_states.to(torch.float32)
    if input_gate is not None:
        assert input_gate.shape == (batch, nheads, nchunks)
        input_gate = input_gate.to(torch.float32)
    if pq_dt is not None:
        assert pq_dt.shape == (batch, nheads, nchunks)
        pq_dt = pq_dt.to(torch.float32)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
        prev_seq_idx = torch.zeros(batch, device=seq_idx.device, dtype=seq_idx.dtype)
    else:
        prev_seq_idx = None

    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=running.dtype)
    out[:, 0] = running
    for c in range(nchunks):
        scale = torch.exp(dA_chunk_cumsum[:, :, c].to(torch.float32))
        if seq_idx is not None:
            seq_idx_new = seq_idx[:, min((c + 1) * chunk_size, seqlen) - 1]
            same_sequence = seq_idx_new == prev_seq_idx
            scale = torch.where(same_sequence[:, None], scale, torch.zeros_like(scale))
            prev_seq_idx = seq_idx_new
        new_states = states[:, c].to(torch.float32)
        running_delta = None
        if pq_P is not None:
            if pq_apply_to in {"running", "both"}:
                running_delta = _compute_low_rank_pq_delta(
                    running,
                    pq_P,
                    pq_Q,
                    pq_alpha,
                    pq_gate=input_gate[:, :, c] if input_gate is not None else None,
                    pq_dt=pq_dt[:, :, c] if pq_dt is not None else None,
                )
            if pq_apply_to in {"new", "both"}:
                new_states = new_states + _compute_low_rank_pq_delta(new_states, pq_P, pq_Q, pq_alpha)
        running = scale[:, :, None] * running + new_states
        if running_delta is not None:
            running = running + running_delta
        if c < nchunks - 1:
            out[:, c + 1] = running
    return out.to(states.dtype), running


def state_passing_ref(states, dA_chunk_cumsum, initial_states=None, input_gate=None):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    if input_gate is None:
        input_gate = torch.ones_like(dA_chunk_cumsum)
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    input_gate = F.pad(input_gate, (1, 0), value=1.0)
    input_gate = torch.cumprod(input_gate, dim=-1)
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    decay_chunk = decay_chunk * (input_gate[:, :, :, None] / input_gate[:, :, None, :])
    causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]
