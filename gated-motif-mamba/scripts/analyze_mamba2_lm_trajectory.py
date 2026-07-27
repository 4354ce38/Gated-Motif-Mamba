#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gated_motif_mamba.paths import DEFAULT_BASE_MODEL, DEFAULT_LOGS_ROOT, DEFAULT_PILE_ROOT

from gated_motif_mamba_ssm.models.config_mamba import MambaConfig
from gated_motif_mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

try:
    from gated_motif_mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn = None, None


class PileSequentialBatcher:
    def __init__(self, pile_root: str):
        pattern = re.compile(r"document-\d{5}-of-\d{5}\.bin$")
        self.files = sorted([p for p in glob(os.path.join(pile_root, "*.bin")) if pattern.search(os.path.basename(p))])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No Pile .bin files found under: {pile_root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.files]
        self.lengths = [int(a.shape[0]) for a in self.arrays]
        self.shard_idx = 0
        self.offset = 0

    def next_batch(self, batch_size: int, seq_len: int) -> torch.Tensor:
        need = seq_len + 1
        out = torch.empty((batch_size, need), dtype=torch.long)
        for i in range(batch_size):
            while True:
                arr = self.arrays[self.shard_idx]
                n = self.lengths[self.shard_idx]
                if self.offset + need <= n:
                    chunk = np.asarray(arr[self.offset:self.offset + need], dtype=np.int64)
                    out[i].copy_(torch.from_numpy(chunk))
                    self.offset += need
                    if self.offset + need > n:
                        self.shard_idx = (self.shard_idx + 1) % len(self.arrays)
                        self.offset = 0
                    break
                self.shard_idx = (self.shard_idx + 1) % len(self.arrays)
                self.offset = 0
        return out


def load_config_local(pretrained_dir: str) -> MambaConfig:
    with open(os.path.join(pretrained_dir, "config.json"), "r", encoding="utf-8") as f:
        return MambaConfig(**json.load(f))


def extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)}")


def infer_state_pq_rank_and_headwise(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, bool]:
    for k, v in state_dict.items():
        if k.endswith("state_pq_Q"):
            if v.ndim == 3:
                return int(v.shape[1]), True
            if v.ndim == 2:
                return int(v.shape[0]), False
    for k, v in state_dict.items():
        if k.endswith("state_pq_P"):
            if v.ndim == 3:
                return int(v.shape[2]), True
            if v.ndim == 2:
                return int(v.shape[1]), False
    return 0, False


def build_model(
    pretrained_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    chunk_size: int,
    input_gate_on_state: bool,
    input_gate_mode: str,
    input_gate_gain: float,
    input_gate_chunk_reduce: str,
    input_gate_chunk_temp: float,
    state_pq_rank: int,
    state_pq_alpha: float,
    state_pq_apply_to: str,
    state_pq_headwise: bool,
) -> MambaLMHeadModel:
    config = load_config_local(pretrained_dir)
    ssm_cfg = dict(config.ssm_cfg or {})
    ssm_cfg["layer"] = "Mamba2"
    ssm_cfg["use_mem_eff_path"] = False
    ssm_cfg["chunk_size"] = int(chunk_size)
    ssm_cfg["input_gate_on_state"] = bool(input_gate_on_state)
    ssm_cfg["input_gate_mode"] = str(input_gate_mode)
    ssm_cfg["input_gate_gain"] = float(input_gate_gain)
    ssm_cfg["input_gate_chunk_reduce"] = str(input_gate_chunk_reduce)
    ssm_cfg["input_gate_chunk_temp"] = float(input_gate_chunk_temp)
    ssm_cfg["state_pq_rank"] = int(state_pq_rank)
    ssm_cfg["state_pq_alpha"] = float(state_pq_alpha)
    ssm_cfg["state_pq_apply_to"] = str(state_pq_apply_to)
    ssm_cfg["state_pq_headwise"] = bool(state_pq_headwise)
    config.ssm_cfg = ssm_cfg
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    base_state = torch.load(os.path.join(pretrained_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(base_state, strict=False)
    model.eval()
    return model


def neutralize_input_gate(model: MambaLMHeadModel) -> None:
    # Force the chunk-path gate to stay extremely close to 1.0 so we can
    # analyze base Mamba2 at chunk granularity without materially changing
    # the recurrence.
    for module in model.modules():
        proj = getattr(module, "input_gate_proj", None)
        if proj is not None:
            with torch.no_grad():
                proj.weight.zero_()
                if getattr(module, "input_gate_mode", "exp_decay") == "exp_decay":
                    proj.bias.fill_(-20.0)
                else:
                    proj.bias.zero_()


def compute_trajectory_length(traj: np.ndarray) -> float:
    if traj.ndim != 2:
        raise ValueError(f"Expected 2D trajectory array, got shape {traj.shape}")
    if traj.shape[0] < 2:
        return 0.0
    arr = traj.astype(np.float64, copy=False)
    finite = np.isfinite(arr)
    if not finite.all():
        finite_vals = np.abs(arr[finite])
        fill = float(finite_vals.max()) if finite_vals.size > 0 else 0.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=fill, neginf=-fill)
    deltas = arr[1:] - arr[:-1]
    # Stable per-step norm: scale first to avoid overflow in sum(delta^2).
    scale = np.max(np.abs(deltas), axis=-1, keepdims=True)
    safe = np.zeros_like(deltas, dtype=np.float64)
    np.divide(deltas, scale, out=safe, where=scale > 0)
    step_norms = np.sqrt(np.sum(safe * safe, axis=-1)) * scale.squeeze(-1)
    return float(step_norms.sum())


def sanitize_for_pca(traj: np.ndarray, normalize: str = "none") -> np.ndarray:
    arr = traj.astype(np.float64, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float64)
    finite_vals = np.abs(arr[finite])
    fill = float(finite_vals.max()) if finite_vals.size > 0 else 0.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=fill, neginf=-fill)
    if normalize == "max_abs":
        max_abs = float(np.max(np.abs(arr)))
        if max_abs > 0:
            arr = arr / max_abs
    elif normalize != "none":
        raise ValueError(f"Unsupported PCA normalization: {normalize}")
    return arr


def pca_project(traj: np.ndarray, normalize: str = "none") -> Tuple[np.ndarray, np.ndarray]:
    if traj.ndim != 2:
        raise ValueError(f"Expected 2D trajectory array, got shape {traj.shape}")
    if traj.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
    safe_traj = sanitize_for_pca(traj, normalize=normalize)
    centered = safe_traj - safe_traj.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    eig = (s ** 2) / max(centered.shape[0] - 1, 1)
    total = eig.sum()
    explained = np.zeros(2, dtype=np.float32)
    if total > 0:
        explained[: min(2, eig.shape[0])] = (eig[:2] / total).astype(np.float32)
    coords2 = np.zeros((traj.shape[0], 2), dtype=np.float32)
    coords2[:, : coords.shape[1]] = coords[:, :2].astype(np.float32)
    return coords2, explained


def plot_trajectory(ax, coords: np.ndarray, title: str, explained: np.ndarray | None = None) -> None:
    if coords.shape[0] == 0:
        ax.set_title(title)
        ax.axis("off")
        return
    ax.plot(coords[:, 0], coords[:, 1], "-o", markersize=2.5, linewidth=1.0)
    ax.scatter(coords[0, 0], coords[0, 1], c="tab:green", s=18, label="start")
    ax.scatter(coords[-1, 0], coords[-1, 1], c="tab:red", s=18, label="end")
    if explained is not None:
        title = f"{title}\nPC1+PC2={100.0 * float(explained.sum()):.1f}%"
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _apply_block_norm(layer, hidden_states, residual):
    if not layer.fused_add_norm:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = layer.norm(residual.to(dtype=layer.norm.weight.dtype))
        if layer.residual_in_fp32:
            residual = residual.to(torch.float32)
        return hidden_states, residual
    hidden_states, residual = layer_norm_fn(
        hidden_states,
        layer.norm.weight,
        layer.norm.bias,
        residual=residual,
        prenorm=True,
        residual_in_fp32=layer.residual_in_fp32,
        eps=layer.norm.eps,
        is_rms_norm=isinstance(layer.norm, RMSNorm),
    )
    return hidden_states, residual


def forward_with_chunk_trajectories(model: MambaLMHeadModel, input_ids: torch.Tensor):
    hidden_states = model.backbone.embedding(input_ids)
    embedding_states = hidden_states.detach().cpu()
    residual = None
    layer_chunk_states: List[torch.Tensor] = []

    for layer in model.backbone.layers:
        hidden_states, residual = _apply_block_norm(layer, hidden_states, residual)
        mixer_out, chunk_states, chunk_final_state = layer.mixer(hidden_states, return_chunk_state=True)
        layer_chunk_states.append(torch.cat([chunk_states, chunk_final_state.unsqueeze(1)], dim=1).detach().cpu())
        hidden_states = mixer_out

        if layer.mlp is not None:
            if not layer.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = layer.norm2(residual.to(dtype=layer.norm2.weight.dtype))
                if layer.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    layer.norm2.weight,
                    layer.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=layer.residual_in_fp32,
                    eps=layer.norm2.eps,
                    is_rms_norm=isinstance(layer.norm2, RMSNorm),
                )
            hidden_states = layer.mlp(hidden_states)

    if not model.backbone.fused_add_norm:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = model.backbone.norm_f(residual.to(dtype=model.backbone.norm_f.weight.dtype))
    else:
        hidden_states = layer_norm_fn(
            hidden_states,
            model.backbone.norm_f.weight,
            model.backbone.norm_f.bias,
            eps=model.backbone.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=model.backbone.residual_in_fp32,
            is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm),
        )
    logits = model.lm_head(hidden_states)
    return embedding_states, layer_chunk_states, hidden_states.detach().cpu(), logits.detach().cpu()


def forward_with_token_ssm_trajectories(model: MambaLMHeadModel, input_ids: torch.Tensor):
    hidden_states = model.backbone.embedding(input_ids)
    embedding_states = hidden_states.detach().cpu()
    residual = None
    layer_ssm_states: List[torch.Tensor] = []

    for layer in model.backbone.layers:
        hidden_states, residual = _apply_block_norm(layer, hidden_states, residual)
        mixer_out, ssm_states = layer.mixer(hidden_states, return_ssm_state=True)
        layer_ssm_states.append(ssm_states.detach().cpu())
        hidden_states = mixer_out

        if layer.mlp is not None:
            if not layer.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = layer.norm2(residual.to(dtype=layer.norm2.weight.dtype))
                if layer.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    layer.norm2.weight,
                    layer.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=layer.residual_in_fp32,
                    eps=layer.norm2.eps,
                    is_rms_norm=isinstance(layer.norm2, RMSNorm),
                )
            hidden_states = layer.mlp(hidden_states)

    if not model.backbone.fused_add_norm:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = model.backbone.norm_f(residual.to(dtype=model.backbone.norm_f.weight.dtype))
    else:
        hidden_states = layer_norm_fn(
            hidden_states,
            model.backbone.norm_f.weight,
            model.backbone.norm_f.bias,
            eps=model.backbone.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=model.backbone.residual_in_fp32,
            is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm),
        )
    logits = model.lm_head(hidden_states)
    return embedding_states, layer_ssm_states, hidden_states.detach().cpu(), logits.detach().cpu()


def forward_with_token_hidden_trajectories(model: MambaLMHeadModel, input_ids: torch.Tensor):
    layer_outputs: List[torch.Tensor] = []
    hooks = []

    def make_hook():
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs.append(hidden.detach().cpu())
        return hook

    for layer in model.backbone.layers:
        hooks.append(layer.register_forward_hook(make_hook()))

    with torch.no_grad():
        embedding_states = model.backbone.embedding(input_ids).detach().cpu()
        hidden_states = model.backbone(input_ids)
        logits = model.lm_head(hidden_states)

    for h in hooks:
        h.remove()

    return embedding_states, layer_outputs, hidden_states.detach().cpu(), logits.detach().cpu()


def chunk_pool_input(hidden: torch.Tensor, chunk_size: int) -> np.ndarray:
    # hidden: [L, D]
    seq_len = hidden.shape[0]
    pooled = []
    for start in range(0, seq_len, chunk_size):
        pooled.append(hidden[start : start + chunk_size].mean(dim=0))
    return torch.stack(pooled, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Plot chunk-level SSM trajectory length for Mamba2 LM models.")
    parser.add_argument("--pretrained-dir", type=str, default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--pile-root", type=str, default=str(DEFAULT_PILE_ROOT))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--input-gate-mode", type=str, default="symmetric_exp", choices=["exp_decay", "symmetric_exp"])
    parser.add_argument("--input-gate-gain", type=float, default=0.5)
    parser.add_argument("--input-gate-chunk-reduce", type=str, default="importance", choices=["last", "importance"])
    parser.add_argument("--input-gate-chunk-temp", type=float, default=4.0)
    parser.add_argument("--state-pq-alpha", type=float, default=1e-2)
    parser.add_argument("--state-pq-apply-to", type=str, default="running", choices=["new", "running", "both"])
    parser.add_argument(
        "--pca-normalize",
        type=str,
        default="none",
        choices=["none", "max_abs"],
        help="Normalization before PCA. Use 'none' for raw PCA scale; 'max_abs' is safer for unstable token-level states.",
    )
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--base-trajectory-level",
        type=str,
        default="chunk",
        choices=["chunk", "token"],
        help="When no adapter is provided, analyze base Mamba2 at chunk-level or token-level.",
    )
    parser.add_argument(
        "--adapter-trajectory-level",
        type=str,
        default="chunk",
        choices=["chunk", "token_hidden", "token_ssm"],
        help="When an adapter is provided, analyze chunk-level SSM trajectory, token-level layer-hidden trajectory, or token-level SSM trajectory.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    use_adapter = bool(args.adapter)
    adapter_state = None
    pq_rank, pq_headwise = 0, False
    if use_adapter:
        adapter_state = extract_state_dict(torch.load(args.adapter, map_location="cpu"))
        pq_rank, pq_headwise = infer_state_pq_rank_and_headwise(adapter_state)

    model = build_model(
        args.pretrained_dir,
        device,
        dtype,
        chunk_size=args.chunk_size,
        input_gate_on_state=use_adapter or args.base_trajectory_level == "chunk",
        input_gate_mode=args.input_gate_mode,
        input_gate_gain=args.input_gate_gain,
        input_gate_chunk_reduce=args.input_gate_chunk_reduce,
        input_gate_chunk_temp=args.input_gate_chunk_temp,
        state_pq_rank=pq_rank,
        state_pq_alpha=args.state_pq_alpha if use_adapter else 0.0,
        state_pq_apply_to=args.state_pq_apply_to,
        state_pq_headwise=pq_headwise,
    )
    if use_adapter:
        model.load_state_dict(adapter_state, strict=False)
    elif args.base_trajectory_level == "chunk":
        neutralize_input_gate(model)
    model.eval()

    batcher = PileSequentialBatcher(args.pile_root)
    batch = batcher.next_batch(args.batch_size, args.seq_len)
    input_ids = batch[:, :-1].to(device)
    sample_idx = int(args.sample_index)

    with torch.no_grad():
        if use_adapter:
            if args.adapter_trajectory_level == "chunk":
                embedding_states, layer_states, _, logits = forward_with_chunk_trajectories(model, input_ids)
                input_traj = chunk_pool_input(embedding_states[sample_idx], args.chunk_size)
                layer_kind = "chunk"
            elif args.adapter_trajectory_level == "token_ssm":
                embedding_states, layer_states, _, logits = forward_with_token_ssm_trajectories(model, input_ids)
                input_traj = embedding_states[sample_idx].numpy()
                layer_kind = "token_ssm"
            else:
                embedding_states, layer_states, _, logits = forward_with_token_hidden_trajectories(model, input_ids)
                input_traj = embedding_states[sample_idx].numpy()
                layer_kind = "token_hidden"
        else:
            if args.base_trajectory_level == "chunk":
                embedding_states, layer_states, _, logits = forward_with_chunk_trajectories(model, input_ids)
                input_traj = chunk_pool_input(embedding_states[sample_idx], args.chunk_size)
                layer_kind = "chunk"
            else:
                embedding_states, layer_states, _, logits = forward_with_token_ssm_trajectories(model, input_ids)
                input_traj = embedding_states[sample_idx].numpy()
                layer_kind = "token"

    if sample_idx >= input_ids.shape[0]:
        raise ValueError(f"sample-index {sample_idx} out of range for batch size {input_ids.shape[0]}")

    if use_adapter:
        adapter_tag = f"{Path(args.adapter).resolve().parent.name}_{args.adapter_trajectory_level}"
    else:
        adapter_tag = f"mamba2_130m_base_{args.base_trajectory_level}"
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_LOGS_ROOT / f"{adapter_tag}_trajectory_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    input_nonfinite = int((~np.isfinite(input_traj)).sum())
    input_max_abs = float(np.nanmax(np.abs(np.nan_to_num(input_traj, nan=0.0, posinf=0.0, neginf=0.0)))) if input_traj.size > 0 else 0.0
    input_coords, input_explained = pca_project(input_traj, normalize=args.pca_normalize)
    rows.append({
        "name": "input",
        "raw_length": compute_trajectory_length(input_traj),
        "pca2_length": compute_trajectory_length(input_coords),
        "pc1_explained": float(input_explained[0]),
        "pc2_explained": float(input_explained[1]),
        "steps": int(input_traj.shape[0]),
        "nonfinite_count": input_nonfinite,
        "max_abs": input_max_abs,
    })

    all_panels = [("Input", input_coords, input_explained)]
    for layer_idx, states in enumerate(layer_states):
        traj = states[sample_idx].reshape(states.shape[1], -1).numpy()
        nonfinite_count = int((~np.isfinite(traj)).sum())
        max_abs = float(np.nanmax(np.abs(np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)))) if traj.size > 0 else 0.0
        coords, explained = pca_project(traj, normalize=args.pca_normalize)
        rows.append({
            "name": f"layer_{layer_idx}",
            "raw_length": compute_trajectory_length(traj),
            "pca2_length": compute_trajectory_length(coords),
            "pc1_explained": float(explained[0]),
            "pc2_explained": float(explained[1]),
            "steps": int(traj.shape[0]),
            "nonfinite_count": nonfinite_count,
            "max_abs": max_abs,
        })
        all_panels.append((f"L{layer_idx}", coords, explained))

    csv_path = out_dir / "trajectory_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    num_panels = len(all_panels)
    cols = 5
    rows_n = int(np.ceil(num_panels / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 3.0))
    axes = np.atleast_1d(axes).reshape(rows_n, cols)
    flat_axes = axes.reshape(-1)
    for ax, (title, coords, explained) in zip(flat_axes, all_panels):
        plot_trajectory(ax, coords, title, explained)
    for ax in flat_axes[len(all_panels):]:
        ax.axis("off")
    top1 = int(logits[sample_idx, -1].argmax().item())
    fig.suptitle(f"{adapter_tag} {layer_kind}-level SSM trajectory analysis | sample={sample_idx} | top1_last={top1}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = out_dir / "trajectory_panels.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    summary_fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5))
    names = [r["name"] for r in rows]
    raw = [r["raw_length"] for r in rows]
    pca2 = [r["pca2_length"] for r in rows]
    x = np.arange(len(names))
    ax.plot(x, raw, marker="o", linewidth=1.5, label="raw length")
    ax.plot(x, pca2, marker="s", linewidth=1.5, label="PCA-2 length")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right")
    ax.set_ylabel("Trajectory Length")
    ax.set_title(f"{layer_kind.capitalize()}-level SSM Trajectory Length Summary")
    ax.legend()
    summary_fig.tight_layout()
    summary_fig.savefig(out_dir / "trajectory_length_summary.png", dpi=180)
    plt.close(summary_fig)

    config_path = out_dir / "analysis_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved trajectory analysis to {out_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"Panels PNG: {fig_path}")


if __name__ == "__main__":
    main()
