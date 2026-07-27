#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors
from matplotlib.lines import Line2D
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gated_motif_mamba.paths import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAIN_ADAPTER,
    DEFAULT_TRUTHFULQA_ADAPTER_ROWS,
    DEFAULT_TRUTHFULQA_BASE_ROWS,
    DEFAULT_TRUTHFULQA_OUT_DIR,
)

from analyze_mamba2_lm_trajectory import forward_with_token_ssm_trajectories
from mc_choice_refusal_benchmark import (
    REFUSAL_TEXT,
    TOKENIZER_PATH,
    TASK_SPECS,
    build_model,
    build_prompt,
    choice_letter,
    load_task_dataset,
)


BASE_MODEL = DEFAULT_BASE_MODEL
ADAPTER_PATH = DEFAULT_MAIN_ADAPTER
BASE_ROWS = DEFAULT_TRUTHFULQA_BASE_ROWS
ADAPTER_ROWS = DEFAULT_TRUTHFULQA_ADAPTER_ROWS
OUT_DIR = DEFAULT_TRUTHFULQA_OUT_DIR

WINDOW = 64
SAMPLES_PER_GROUP = 8
TOP_LAYERS = 6
CASE_TRAJ_WINDOW = 96
CASE_LAYER = -1
PQ_COMPONENTS = (0, 1)
PROFILE_TAIL = 24
MULTI_CASES_PER_GROUP = 2
FILMSTRIP_FRAMES = 8
FILMSTRIP_HEAD_FRAMES = 4
FILMSTRIP_TAIL_FRAMES = 4
FIXED_TRAJ_TICK = 0.1
FIXED_TRAJ_PAD = 0.015
TRANSITION_GRID = (
    ("wrong", "wrong"),
    ("wrong", "refusal"),
    ("wrong", "correct"),
    ("refusal", "wrong"),
    ("refusal", "refusal"),
    ("refusal", "correct"),
    ("correct", "wrong"),
    ("correct", "refusal"),
    ("correct", "correct"),
)
TRANSITION_GRID_OTHERS = tuple(pair for pair in TRANSITION_GRID if pair not in {("wrong", "refusal"), ("wrong", "correct")})


def load_transition_table() -> pd.DataFrame:
    base = pd.read_csv(BASE_ROWS).rename(
        columns={
            "category": "base_cat",
            "predicted_answer": "base_pred",
            "gold_score": "base_gold_score",
            "refusal_score": "base_refusal_score",
            "top_score": "base_top_score",
        }
    )
    adapter = pd.read_csv(ADAPTER_ROWS).rename(
        columns={
            "category": "adapter_cat",
            "predicted_answer": "adapter_pred",
            "gold_score": "adapter_gold_score",
            "refusal_score": "adapter_refusal_score",
            "top_score": "adapter_top_score",
        }
    )
    merged = base.merge(
        adapter[
            [
                "question_id",
                "adapter_cat",
                "adapter_pred",
                "adapter_gold_score",
                "adapter_refusal_score",
                "adapter_top_score",
            ]
        ],
        on="question_id",
        how="inner",
    )
    return merged.sort_values("question_id").reset_index(drop=True)


def select_group_question_ids(merged: pd.DataFrame, base_cat: str, adapter_cat: str, limit: int) -> List[int]:
    subset = merged[(merged.base_cat == base_cat) & (merged.adapter_cat == adapter_cat)].copy()
    if adapter_cat == "correct":
        subset["rank_score"] = subset["adapter_gold_score"] - subset["base_gold_score"]
        subset = subset.sort_values("rank_score", ascending=False)
    elif adapter_cat == "refusal":
        subset["rank_score"] = subset["adapter_refusal_score"] - subset["base_refusal_score"]
        subset = subset.sort_values("rank_score", ascending=False)
    else:
        subset["rank_score"] = subset["adapter_top_score"] - subset["base_top_score"]
        subset = subset.sort_values("rank_score", ascending=False)
    return subset["question_id"].head(limit).astype(int).tolist()


def select_case_question_id(merged: pd.DataFrame, base_cat: str, adapter_cat: str) -> int:
    ids = select_group_question_ids(merged, base_cat, adapter_cat, limit=1)
    if not ids:
        raise RuntimeError(f"No samples found for transition {base_cat}->{adapter_cat}")
    return ids[0]


def load_truthfulqa_examples() -> Sequence[object]:
    spec = TASK_SPECS["truthfulqa_mc1"]
    return load_task_dataset(spec, limit=None)


def build_prompt_and_targets(dataset: Sequence[object], question_id: int) -> Dict[str, object]:
    spec = TASK_SPECS["truthfulqa_mc1"]
    example = spec.extractor(dataset[int(question_id)], int(question_id))
    prompt = build_prompt(example.question, example.choices)
    gold_letter = choice_letter(example.gold_index)
    gold_candidate = f" {gold_letter}"
    refusal_candidate = REFUSAL_TEXT
    return {
        "prompt": prompt,
        "question": example.question,
        "choices": example.choices,
        "gold_index": example.gold_index,
        "gold_letter": gold_letter,
        "gold_candidate": gold_candidate,
        "refusal_candidate": refusal_candidate,
    }


def candidate_text_for_category(obj: Dict[str, object], category: str, predicted_answer: str | None) -> str:
    if category == "correct":
        return str(obj["gold_candidate"])
    if category == "refusal":
        return str(obj["refusal_candidate"])
    pred = "" if predicted_answer is None else str(predicted_answer).strip()
    if not pred or pred.upper() == "REFUSAL":
        return str(obj["refusal_candidate"])
    return f" {pred}"


def fixed_plot_limits() -> Tuple[Tuple[float, float], Tuple[float, float]]:
    bound = FIXED_TRAJ_TICK + FIXED_TRAJ_PAD
    return (-bound, bound), (-bound, bound)


def fixed_plot_limits_for_base_panels(limit: float = 0.08, pad: float = 0.01) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    bound = limit + pad
    return (-bound, bound), (-bound, bound)


def compute_square_coord_limits(
    coord_sets: Sequence[np.ndarray],
    pad_ratio: float = 0.12,
    round_step: float = 0.01,
    min_bound: float = 0.03,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    stacked = np.concatenate(coord_sets, axis=0)
    finite = stacked[np.isfinite(stacked).all(axis=1)]
    if finite.size == 0:
        bound = min_bound
    else:
        max_abs = float(np.max(np.abs(finite)))
        raw_bound = max(min_bound, max_abs * (1.0 + pad_ratio))
        bound = round_step * math.ceil(raw_bound / round_step)
    return (-bound, bound), (-bound, bound), bound


def compute_rounded_coord_limits(
    coord_sets: Sequence[np.ndarray],
    pad_ratio_x: float = 0.08,
    pad_ratio_y: float = 0.08,
    round_step: float = 0.01,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    stacked = np.concatenate(coord_sets, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    xlim = (mins[0] - spans[0] * pad_ratio_x, maxs[0] + spans[0] * pad_ratio_x)
    ylim = (mins[1] - spans[1] * pad_ratio_y, maxs[1] + spans[1] * pad_ratio_y)
    xmin = round_step * math.floor(xlim[0] / round_step)
    xmax = round_step * math.ceil(xlim[1] / round_step)
    ymin = round_step * math.floor(ylim[0] / round_step)
    ymax = round_step * math.ceil(ylim[1] / round_step)
    return (xmin, xmax), (ymin, ymax)


def tokenize_text(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return ids


def run_token_ssm(model, tokenizer, text: str, device: torch.device):
    input_ids = tokenize_text(tokenizer, text, device)
    with torch.no_grad():
        _, layer_states, hidden, logits = forward_with_token_ssm_trajectories(model, input_ids)
    return input_ids[0].detach().cpu(), layer_states, hidden, logits


def ssm_rms_diff_matrix(base_states: List[torch.Tensor], adapter_states: List[torch.Tensor]) -> np.ndarray:
    mats = []
    for base_layer, adapter_layer in zip(base_states, adapter_states):
        diff = adapter_layer[0].float() - base_layer[0].float()  # [L,H,P,N]
        rms = torch.sqrt(torch.mean(diff * diff, dim=(1, 2, 3)))
        mats.append(rms.detach().cpu().numpy())
    return np.stack(mats, axis=0).astype(np.float32)


def align_last_window(mat: np.ndarray, window: int) -> np.ndarray:
    out = np.full((mat.shape[0], window), np.nan, dtype=np.float32)
    take = min(window, mat.shape[1])
    out[:, -take:] = mat[:, -take:]
    return out


def average_aligned_mats(mats: Iterable[np.ndarray], window: int) -> np.ndarray:
    aligned = [align_last_window(mat, window) for mat in mats]
    return np.nanmean(np.stack(aligned, axis=0), axis=0)


def compute_pq_delta_norm_matrix(model, layer_states: List[torch.Tensor]) -> np.ndarray:
    mats = []
    for layer, states in zip(model.backbone.layers, layer_states):
        mixer = layer.mixer
        if getattr(mixer, "state_pq_P", None) is None or getattr(mixer, "state_pq_Q", None) is None:
            rms = torch.zeros(states.shape[1], dtype=torch.float32)
        else:
            state = states[0].float()  # [L,H,P,N]
            state_flat = state.reshape(state.shape[0], state.shape[1], -1)
            P = mixer.state_pq_P.detach().cpu().float()
            Q = mixer.state_pq_Q.detach().cpu().float()
            if P.dim() == 2:
                state_low = torch.einsum("thd,dr->thr", state_flat, P)
                state_mix = torch.einsum("thr,rd->thd", state_low, Q)
            else:
                state_low = torch.einsum("thd,hdr->thr", state_flat, P)
                state_mix = torch.einsum("thr,hrd->thd", state_low, Q)
            pq_delta = float(mixer.state_pq_alpha) * torch.tanh(state_mix)
            rms = torch.sqrt(torch.mean(pq_delta * pq_delta, dim=(1, 2)))
        mats.append(rms.detach().cpu().numpy())
    return np.stack(mats, axis=0).astype(np.float32)


def compute_state_jump_norm_matrix(layer_states: List[torch.Tensor]) -> np.ndarray:
    mats = []
    for states in layer_states:
        state = states[0].float()
        jump = state[1:] - state[:-1]
        rms = torch.sqrt(torch.mean(jump * jump, dim=(1, 2, 3)))
        padded = torch.cat([torch.zeros(1, dtype=rms.dtype), rms], dim=0)
        mats.append(padded.detach().cpu().numpy())
    return np.stack(mats, axis=0).astype(np.float32)


def top_layer_mean_curve(mat: np.ndarray, top_layers: int) -> np.ndarray:
    start = max(0, mat.shape[0] - top_layers)
    return np.nanmean(mat[start:], axis=0)


def tail_mean_per_layer(mat: np.ndarray, tail: int) -> np.ndarray:
    take = min(tail, mat.shape[1])
    return np.nanmean(mat[:, -take:], axis=1)


def final_layer_pq_coords(adapter_model, states: List[torch.Tensor], layer_index: int, components: Tuple[int, int]) -> np.ndarray:
    layer_idx = layer_index if layer_index >= 0 else len(states) + layer_index
    mixer = adapter_model.backbone.layers[layer_idx].mixer
    state = states[layer_idx][0].float()
    state_flat = state.reshape(state.shape[0], state.shape[1], -1)
    P = mixer.state_pq_P.detach().cpu().float()
    if P.dim() != 2:
        raise NotImplementedError("Expected shared-in-block PQ (2D P) for 130m adapter.")
    coords = []
    for comp in components:
        coord = torch.einsum("thd,d->th", state_flat, P[:, comp]).mean(dim=1)
        coords.append(coord)
    return torch.stack(coords, dim=-1).numpy().astype(np.float32)


def draw_heatmap(ax, data: np.ndarray, title: str) -> None:
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Token position from sequence end", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ticks = np.linspace(0, data.shape[1] - 1, 5, dtype=int)
    labels = [str(t - (data.shape[1] - 1)) for t in ticks]
    ax.set_xticks(ticks, labels)
    ax.tick_params(labelsize=8)
    return im


def draw_case_panel(
    ax,
    base_coords: np.ndarray,
    adapter_coords: np.ndarray,
    title: str,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    show_background: bool = True,
) -> object | None:
    bg_im = None
    if show_background and xlim is not None and ylim is not None:
        bg_im = draw_empirical_flow_field(ax, [base_coords, adapter_coords], xlim, ylim)
    ax.plot(base_coords[:, 0], base_coords[:, 1], color="#ff7f0e", linewidth=0.6, alpha=0.84, label="mamba")
    ax.plot(adapter_coords[:, 0], adapter_coords[:, 1], color="#1f77b4", linewidth=0.6, alpha=0.84, label="motifmamba")
    ax.scatter(base_coords[0, 0], base_coords[0, 1], color="#ff7f0e", s=34, marker="o", zorder=3)
    ax.scatter(adapter_coords[0, 0], adapter_coords[0, 1], color="#1f77b4", s=34, marker="o", zorder=3)
    ax.scatter(base_coords[-1, 0], base_coords[-1, 1], color="#ff7f0e", s=30, marker="s", zorder=3)
    ax.scatter(adapter_coords[-1, 0], adapter_coords[-1, 1], color="#1f77b4", s=30, marker="s", zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PQ component 0", fontsize=9)
    ax.set_ylabel("PQ component 1", fontsize=9)
    ax.grid(False)
    ax.axhline(0.0, color="#5f5f5f", linewidth=0.8, alpha=0.42, zorder=1)
    ax.axvline(0.0, color="#5f5f5f", linewidth=0.8, alpha=0.42, zorder=1)
    ax.tick_params(labelsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return bg_im


def estimate_flow_grid(
    coord_sets: Sequence[np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_size: int = 33,
    sigma_frac: float = 0.24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mids = []
    deltas = []
    for coords in coord_sets:
        if len(coords) < 2:
            continue
        mids.append(0.5 * (coords[:-1] + coords[1:]))
        deltas.append(coords[1:] - coords[:-1])
    if not mids:
        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(x, y)
        zeros = np.zeros_like(X)
        support = np.zeros_like(X)
        return x, y, zeros, zeros, support

    mids = np.concatenate(mids, axis=0)
    deltas = np.concatenate(deltas, axis=0)
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)
    sigma = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * sigma_frac
    sigma2 = max(sigma * sigma, 1e-6)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    support = np.zeros_like(X)
    for idx in range(mids.shape[0]):
        dx = X - mids[idx, 0]
        dy = Y - mids[idx, 1]
        w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma2))
        U += w * deltas[idx, 0]
        V += w * deltas[idx, 1]
        support += w
    support_safe = np.maximum(support, 1e-6)
    U = U / support_safe
    V = V / support_safe
    global_delta = np.mean(deltas, axis=0)
    edge_blend = np.exp(-support / (0.28 * np.max(support) + 1e-6))
    U = (1.0 - edge_blend) * U + edge_blend * global_delta[0]
    V = (1.0 - edge_blend) * V + edge_blend * global_delta[1]
    return x, y, U, V, support


def draw_empirical_flow_field(
    ax,
    coord_sets: Sequence[np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    x, y, U, V, support = estimate_flow_grid(coord_sets, xlim, ylim)
    if np.max(support) <= 0:
        return
    density_norm = support / (np.max(support) + 1e-6)
    line_cmap = colors.LinearSegmentedColormap.from_list(
        "green_yellow_density",
        [
            (0.0, (0.12, 0.62, 0.34, 1.0)),
            (0.55, (0.45, 0.76, 0.24, 1.0)),
            (1.0, (0.96, 0.84, 0.16, 1.0)),
        ],
    )
    stream = ax.streamplot(
        x,
        y,
        U,
        V,
        density=1.05,
        color=density_norm,
        cmap=line_cmap,
        linewidth=0.58 + 0.92 * density_norm,
        arrowsize=0.62,
        minlength=0.06,
        maxlength=3.2,
        integration_direction="both",
        zorder=-2,
    )
    stream.lines.set_alpha(1.0)
    stream.arrows.set_alpha(1.0)
    return stream.lines


def compute_coord_limits(coord_sets: Sequence[np.ndarray], pad_ratio: float = 0.08) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    stacked = np.concatenate(coord_sets, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    pads = spans * pad_ratio
    return (mins[0] - pads[0], maxs[0] + pads[0]), (mins[1] - pads[1], maxs[1] + pads[1])


def coords_excess(coords: np.ndarray, limit: float) -> float:
    return float(np.max(np.abs(coords)) - limit)


def choose_bounded_transition_case(
    merged: pd.DataFrame,
    dataset: Sequence[object],
    tokenizer,
    device: torch.device,
    base_model,
    adapter_model,
    base_cat: str,
    adapter_cat: str,
    limit: float = FIXED_TRAJ_TICK,
) -> Tuple[int, np.ndarray, np.ndarray]:
    ranked_ids = select_group_question_ids(merged, base_cat, adapter_cat, limit=10_000)
    best_item: Tuple[float, int, np.ndarray, np.ndarray] | None = None
    for qid in ranked_ids:
        row = merged.loc[merged.question_id == qid].iloc[0]
        obj = build_prompt_and_targets(dataset, qid)
        candidate = candidate_text_for_category(obj, adapter_cat, row["adapter_pred"])
        full_text = obj["prompt"] + candidate
        _, base_states, _, _ = run_token_ssm(base_model, tokenizer, full_text, device)
        _, adapter_states, _, _ = run_token_ssm(adapter_model, tokenizer, full_text, device)
        base_coords = final_layer_pq_coords(adapter_model, base_states, CASE_LAYER, PQ_COMPONENTS)[-CASE_TRAJ_WINDOW:]
        adapter_coords = final_layer_pq_coords(adapter_model, adapter_states, CASE_LAYER, PQ_COMPONENTS)[-CASE_TRAJ_WINDOW:]
        overflow = max(coords_excess(base_coords, limit), coords_excess(adapter_coords, limit))
        if best_item is None or overflow < best_item[0]:
            best_item = (overflow, int(qid), base_coords, adapter_coords)
        if overflow <= 0:
            return int(qid), base_coords, adapter_coords
    if best_item is None:
        raise RuntimeError(f"No samples found for transition {base_cat}->{adapter_cat}")
    return best_item[1], best_item[2], best_item[3]


def draw_model_case_panel(
    ax,
    coords: np.ndarray,
    title: str,
    color: str,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    show_background: bool = True,
) -> object | None:
    bg_im = None
    if show_background and xlim is not None and ylim is not None:
        bg_im = draw_empirical_flow_field(ax, [coords], xlim, ylim)
    ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=0.6, alpha=0.96)
    ax.scatter(coords[0, 0], coords[0, 1], color=color, s=54, marker="o", edgecolors="black", linewidths=0.7, zorder=3)
    ax.scatter(coords[-1, 0], coords[-1, 1], color=color, s=48, marker="s", edgecolors="black", linewidths=0.7, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PQ component 0", fontsize=9)
    ax.set_ylabel("PQ component 1", fontsize=9)
    ax.grid(False)
    ax.axhline(0.0, color="#5f5f5f", linewidth=0.8, alpha=0.42, zorder=1)
    ax.axvline(0.0, color="#5f5f5f", linewidth=0.8, alpha=0.42, zorder=1)
    ax.tick_params(labelsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return bg_im


def sample_frame_endpoints(length: int, n_frames: int) -> List[int]:
    if length <= 1:
        return [length]
    raw = np.linspace(1, length, n_frames + 1)[1:]
    endpoints: List[int] = []
    prev = 1
    for value in raw:
        idx = max(prev + 1, int(round(float(value))))
        idx = min(idx, length)
        endpoints.append(idx)
        prev = idx
    endpoints[-1] = length
    return endpoints


def condensed_frame_specs(
    length: int,
    n_frames: int,
    head_frames: int,
    tail_frames: int,
) -> List[Tuple[str, int | None]]:
    del n_frames
    if length <= head_frames + tail_frames:
        return [(f"{end_idx}/{length}", end_idx) for end_idx in range(1, length + 1)]
    head = list(range(1, head_frames + 1))
    tail = list(range(length - tail_frames + 1, length + 1))
    specs: List[Tuple[str, int | None]] = []
    specs.extend((f"{end_idx}/{length}", end_idx) for end_idx in head)
    specs.append(("...", None))
    specs.extend((f"{end_idx}/{length}", end_idx) for end_idx in tail)
    return specs


def tail_frame_specs(length: int, tail_frames: int) -> List[Tuple[str, int]]:
    start = max(1, length - tail_frames + 1)
    return [(f"{end_idx}/{length}", end_idx) for end_idx in range(start, length + 1)]


def draw_filmstrip_frame_panel(
    ax,
    coords: np.ndarray,
    end_idx: int | None,
    color: str,
    frame_label: str,
    row_label: str | None = None,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    if end_idx is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, frame_label, transform=ax.transAxes, ha="center", va="center", fontsize=16, color="#777777")
        if row_label:
            ax.text(
                0.03,
                0.97,
                row_label,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.2,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
            )
        return
    end_idx = max(1, min(end_idx, len(coords)))
    prefix = coords[:end_idx]
    draw_empirical_flow_field(ax, [prefix], xlim, ylim)
    ax.plot(coords[:, 0], coords[:, 1], color="#d7d7d7", linewidth=0.35, alpha=0.8, zorder=1)
    ax.plot(prefix[:, 0], prefix[:, 1], color=color, linewidth=0.75, alpha=0.96, zorder=2)
    if len(prefix) >= 2:
        ax.plot(prefix[-2:, 0], prefix[-2:, 1], color="#d62728", linewidth=1.1, alpha=1.0, zorder=3)
    ax.scatter(coords[0, 0], coords[0, 1], color=color, s=38, marker="o", edgecolors="black", linewidths=0.6, zorder=3)
    ax.scatter(prefix[-1, 0], prefix[-1, 1], color=color, s=42, marker="s", edgecolors="black", linewidths=0.7, zorder=4)
    ax.axhline(0.0, color="#5f5f5f", linewidth=0.7, alpha=0.35, zorder=0)
    ax.axvline(0.0, color="#5f5f5f", linewidth=0.7, alpha=0.35, zorder=0)
    ax.grid(False)
    ax.set_title(frame_label, fontsize=8.5)
    ax.tick_params(labelsize=7)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if row_label:
        ax.text(
            0.03,
            0.97,
            row_label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
        )


def draw_layer_profile_panel(
    ax,
    wc_mat: np.ndarray,
    wr_mat: np.ndarray,
    title: str,
    tail: int,
    x_label: str = "Layer",
) -> None:
    layers = np.arange(wc_mat.shape[0])
    ax.plot(layers, tail_mean_per_layer(wc_mat, tail), color="#1f77b4", linewidth=2.0, label="wrong->correct")
    ax.plot(layers, tail_mean_per_layer(wr_mat, tail), color="#d62728", linewidth=2.0, label="wrong->refusal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Mean over answer tail", fontsize=9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))

    merged = load_transition_table()
    group_wc = select_group_question_ids(merged, "wrong", "correct", SAMPLES_PER_GROUP)
    group_wr = select_group_question_ids(merged, "wrong", "refusal", SAMPLES_PER_GROUP)
    case_wc = select_case_question_id(merged, "wrong", "correct")
    case_wr = select_case_question_id(merged, "wrong", "refusal")
    multi_wc = select_group_question_ids(merged, "wrong", "correct", MULTI_CASES_PER_GROUP)
    multi_wr = select_group_question_ids(merged, "wrong", "refusal", MULTI_CASES_PER_GROUP)

    print("Selected wrong->correct ids:", group_wc)
    print("Selected wrong->refusal ids:", group_wr)
    print("Case study wrong->correct:", case_wc)
    print("Case study wrong->refusal:", case_wr)

    base_model = build_model(BASE_MODEL, None, device, dtype)
    adapter_model = build_model(BASE_MODEL, ADAPTER_PATH, device, dtype)
    dataset = load_truthfulqa_examples()

    wc_diff_mats = []
    wr_diff_mats = []
    wc_pq_mats = []
    wr_pq_mats = []
    wc_jump_mats = []
    wr_jump_mats = []

    for qid in group_wc:
        obj = build_prompt_and_targets(dataset, qid)
        prompt = obj["prompt"]
        _, base_states, _, _ = run_token_ssm(base_model, tokenizer, prompt, device)
        _, adapter_states, _, _ = run_token_ssm(adapter_model, tokenizer, prompt, device)
        wc_diff_mats.append(ssm_rms_diff_matrix(base_states, adapter_states))
        wc_pq_mats.append(compute_pq_delta_norm_matrix(adapter_model, adapter_states))
        wc_jump_mats.append(compute_state_jump_norm_matrix(adapter_states))

    for qid in group_wr:
        obj = build_prompt_and_targets(dataset, qid)
        prompt = obj["prompt"]
        _, base_states, _, _ = run_token_ssm(base_model, tokenizer, prompt, device)
        _, adapter_states, _, _ = run_token_ssm(adapter_model, tokenizer, prompt, device)
        wr_diff_mats.append(ssm_rms_diff_matrix(base_states, adapter_states))
        wr_pq_mats.append(compute_pq_delta_norm_matrix(adapter_model, adapter_states))
        wr_jump_mats.append(compute_state_jump_norm_matrix(adapter_states))

    wc_diff = average_aligned_mats(wc_diff_mats, WINDOW)
    wr_diff = average_aligned_mats(wr_diff_mats, WINDOW)
    wc_pq = average_aligned_mats(wc_pq_mats, WINDOW)
    wr_pq = average_aligned_mats(wr_pq_mats, WINDOW)
    wc_jump = average_aligned_mats(wc_jump_mats, WINDOW)
    wr_jump = average_aligned_mats(wr_jump_mats, WINDOW)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    im0 = draw_heatmap(axes[0], wc_diff, "wrong -> correct\nmean ||adapter state - base state||")
    im1 = draw_heatmap(axes[1], wr_diff, "wrong -> refusal\nmean ||adapter state - base state||")
    cbar = fig.colorbar(im1, ax=axes, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "truthfulqa_state_diff_heatmaps.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "truthfulqa_state_diff_heatmaps.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    im0 = draw_heatmap(axes[0], wc_pq, "wrong -> correct\nmean PQ delta norm")
    im1 = draw_heatmap(axes[1], wr_pq, "wrong -> refusal\nmean PQ delta norm")
    cbar = fig.colorbar(im1, ax=axes, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "truthfulqa_pq_delta_heatmaps.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "truthfulqa_pq_delta_heatmaps.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    im0 = draw_heatmap(axes[0], wc_jump, "wrong -> correct\nmean state jump norm")
    im1 = draw_heatmap(axes[1], wr_jump, "wrong -> refusal\nmean state jump norm")
    cbar = fig.colorbar(im1, ax=axes, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "truthfulqa_state_jump_heatmaps.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "truthfulqa_state_jump_heatmaps.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11.5), sharex=True, sharey="row")
    images = []
    images.append(draw_heatmap(axes[0, 0], wc_diff, "wrong -> correct\nstate diff"))
    images.append(draw_heatmap(axes[0, 1], wr_diff, "wrong -> refusal\nstate diff"))
    images.append(draw_heatmap(axes[1, 0], wc_pq, "wrong -> correct\nPQ delta"))
    images.append(draw_heatmap(axes[1, 1], wr_pq, "wrong -> refusal\nPQ delta"))
    images.append(draw_heatmap(axes[2, 0], wc_jump, "wrong -> correct\nstate jump"))
    images.append(draw_heatmap(axes[2, 1], wr_jump, "wrong -> refusal\nstate jump"))
    for row in range(3):
        cbar = fig.colorbar(images[2 * row + 1], ax=axes[row, :], fraction=0.020, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "truthfulqa_internal_heatmap_grid.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "truthfulqa_internal_heatmap_grid.svg", bbox_inches="tight")
    plt.close(fig)

    x = np.arange(WINDOW) - (WINDOW - 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    axes[0].plot(x, top_layer_mean_curve(wc_pq, TOP_LAYERS), color="#1f77b4", linewidth=2.0, label="wrong->correct")
    axes[0].plot(x, top_layer_mean_curve(wr_pq, TOP_LAYERS), color="#d62728", linewidth=2.0, label="wrong->refusal")
    axes[0].set_title(f"Mean PQ delta norm (top {TOP_LAYERS} layers)")
    axes[0].set_xlabel("Token position from sequence end")
    axes[0].set_ylabel("Norm")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.6)
    axes[0].legend()

    axes[1].plot(x, top_layer_mean_curve(wc_jump, TOP_LAYERS), color="#1f77b4", linewidth=2.0, label="wrong->correct")
    axes[1].plot(x, top_layer_mean_curve(wr_jump, TOP_LAYERS), color="#d62728", linewidth=2.0, label="wrong->refusal")
    axes[1].set_title(f"Mean state jump norm (top {TOP_LAYERS} layers)")
    axes[1].set_xlabel("Token position from sequence end")
    axes[1].set_ylabel("Norm")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "truthfulqa_pqdelta_statejump_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "truthfulqa_pqdelta_statejump_curves.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True)
    draw_layer_profile_panel(
        axes[0],
        wc_diff,
        wr_diff,
        f"Layer profile: state diff\n(last {PROFILE_TAIL} tokens)",
        PROFILE_TAIL,
    )
    draw_layer_profile_panel(
        axes[1],
        wc_pq,
        wr_pq,
        f"Layer profile: PQ delta\n(last {PROFILE_TAIL} tokens)",
        PROFILE_TAIL,
    )
    draw_layer_profile_panel(
        axes[2],
        wc_jump,
        wr_jump,
        f"Layer profile: state jump\n(last {PROFILE_TAIL} tokens)",
        PROFILE_TAIL,
    )
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "truthfulqa_layer_profiles.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "truthfulqa_layer_profiles.svg", bbox_inches="tight")
    plt.close(fig)

    case_specs = [
        (case_wr, "wrong -> refusal", "refusal"),
        (case_wc, "wrong -> correct", "gold"),
    ]
    case_panels = []
    for qid, title, candidate_type in case_specs:
        obj = build_prompt_and_targets(dataset, qid)
        candidate = obj["gold_candidate"] if candidate_type == "gold" else obj["refusal_candidate"]
        full_text = obj["prompt"] + candidate
        prompt_ids = tokenize_text(tokenizer, obj["prompt"], device).shape[1]
        split_at = max(1, prompt_ids - 1)
        _, base_states, _, _ = run_token_ssm(base_model, tokenizer, full_text, device)
        _, adapter_states, _, _ = run_token_ssm(adapter_model, tokenizer, full_text, device)
        base_coords = final_layer_pq_coords(adapter_model, base_states, CASE_LAYER, PQ_COMPONENTS)
        adapter_coords = final_layer_pq_coords(adapter_model, adapter_states, CASE_LAYER, PQ_COMPONENTS)
        base_coords = base_coords[-CASE_TRAJ_WINDOW:]
        adapter_coords = adapter_coords[-CASE_TRAJ_WINDOW:]
        case_panels.append((qid, title, base_coords, adapter_coords))
    xlim, ylim = compute_coord_limits(
        [item[2] for item in case_panels] + [item[3] for item in case_panels]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    bg_im = None
    for ax, (qid, title, base_coords, adapter_coords) in zip(axes, case_panels):
        bg_im = draw_case_panel(ax, base_coords, adapter_coords, f"{title}\nqid={qid}", xlim=xlim, ylim=ylim)
    legend_handles = [
        Line2D([0], [0], color="#ff7f0e", linewidth=0.6, label="mamba"),
        Line2D([0], [0], color="#1f77b4", linewidth=0.6, label="motifmamba"),
        Line2D([0], [0], color="#444444", marker="o", linestyle="None", markersize=4.8, label="trajectory start"),
        Line2D([0], [0], color="#444444", marker="s", linestyle="None", markersize=4.4, label="trajectory end"),
    ]
    if bg_im is not None:
        cbar = fig.colorbar(bg_im, ax=axes, fraction=0.024, pad=0.02)
        cbar.set_label("field density", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_DIR / "truthfulqa_case_pq_trajectories.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "truthfulqa_case_pq_trajectories.svg", bbox_inches="tight")
    plt.close(fig)

    base_panel_limit = 0.08
    grouped_panels_by_base: Dict[str, List[Tuple[int, str, np.ndarray, np.ndarray]]] = {}
    all_base_panel_coords: List[np.ndarray] = []
    for base_cat in ("wrong", "refusal", "correct"):
        grouped_panels = []
        for adapter_cat in ("wrong", "refusal", "correct"):
            qid, base_coords, adapter_coords = choose_bounded_transition_case(
                merged,
                dataset,
                tokenizer,
                device,
                base_model,
                adapter_model,
                base_cat,
                adapter_cat,
                limit=base_panel_limit,
            )
            grouped_panels.append((qid, f"{base_cat} -> {adapter_cat}", base_coords, adapter_coords))
            all_base_panel_coords.extend([base_coords, adapter_coords])
        grouped_panels_by_base[base_cat] = grouped_panels

    base_global_xlim, base_global_ylim = compute_rounded_coord_limits(
        all_base_panel_coords,
        pad_ratio_x=0.06,
        pad_ratio_y=0.02,
        round_step=0.01,
    )

    for base_cat in ("wrong", "refusal", "correct"):
        grouped_panels = grouped_panels_by_base[base_cat]
        fig, axes = plt.subplots(len(grouped_panels), 2, figsize=(11.5, 3.7 * len(grouped_panels)), sharex=True, sharey=True)
        if len(grouped_panels) == 1:
            axes = np.expand_dims(axes, axis=0)
        bg_im = None
        for row, (qid, title, base_coords, adapter_coords) in enumerate(grouped_panels):
            bg_im = draw_model_case_panel(
                axes[row, 0],
                base_coords,
                "",
                "#ff7f0e",
                xlim=base_global_xlim,
                ylim=base_global_ylim,
                show_background=True,
            )
            bg_im = draw_model_case_panel(
                axes[row, 1],
                adapter_coords,
                "",
                "#1f77b4",
                xlim=base_global_xlim,
                ylim=base_global_ylim,
                show_background=True,
            )
        for ax in axes.ravel():
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([base_global_xlim[0], base_global_xlim[1]])
            ax.set_yticks([base_global_ylim[0], base_global_ylim[1]])
            ax.set_xticklabels([f"{base_global_xlim[0]:.2f}", f"{base_global_xlim[1]:.2f}"])
            ax.set_yticklabels([f"{base_global_ylim[0]:.2f}", f"{base_global_ylim[1]:.2f}"])
            ax.tick_params(length=2.5, width=0.7, labelbottom=True, labelleft=True, labelsize=8)
        fig.tight_layout(rect=(0, 0, 1, 1))
        stem = f"truthfulqa_case_pq_trajectories_{base_cat}_base"
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    motif_frame_specs = tail_frame_specs(case_panels[0][3].shape[0], FILMSTRIP_TAIL_FRAMES)
    fig, axes = plt.subplots(
        len(motif_frame_specs),
        len(case_panels),
        figsize=(7.2, 2.25 * len(motif_frame_specs)),
        sharex=True,
        sharey=True,
    )
    if len(motif_frame_specs) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(case_panels) == 1:
        axes = np.expand_dims(axes, axis=1)
    for row, (frame_label, end_idx) in enumerate(motif_frame_specs):
        for col, (qid, title, _, adapter_coords) in enumerate(case_panels):
            draw_filmstrip_frame_panel(
                axes[row, col],
                adapter_coords,
                end_idx,
                "#1f77b4",
                frame_label,
                row_label=f"{title}\nqid={qid}" if row == 0 else None,
                xlim=xlim,
                ylim=ylim,
            )
    for ax in axes[-1]:
        ax.set_xlabel("PQ component 0", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("PQ component 1", fontsize=8)
    filmstrip_legend = [
        Line2D([0], [0], color="#d7d7d7", linewidth=0.6, label="full trajectory"),
        Line2D([0], [0], color="#1f77b4", linewidth=0.75, label="motifmamba prefix"),
        Line2D([0], [0], color="#1f77b4", marker="o", markeredgecolor="black", linestyle="None", markersize=5.0, label="start"),
        Line2D([0], [0], color="#1f77b4", marker="s", markeredgecolor="black", linestyle="None", markersize=5.0, label="current frame end"),
    ]
    fig.legend(filmstrip_legend, [h.get_label() for h in filmstrip_legend], loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_DIR / "truthfulqa_case_pq_motifmamba_filmstrip.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "truthfulqa_case_pq_motifmamba_filmstrip.svg", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "wrong_to_correct_ids": group_wc,
        "wrong_to_refusal_ids": group_wr,
        "case_wrong_to_correct": case_wc,
        "case_wrong_to_refusal": case_wr,
        "multi_wrong_to_correct": multi_wc,
        "multi_wrong_to_refusal": multi_wr,
        "window": WINDOW,
        "samples_per_group": SAMPLES_PER_GROUP,
        "top_layers": TOP_LAYERS,
    }
    (OUT_DIR / "manifest.json").write_text(pd.Series(manifest).to_json(indent=2), encoding="utf-8")
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
