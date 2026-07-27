from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class AdapterArchitecture:
    checkpoint_path: Path
    state_pq_rank: int
    state_pq_headwise: bool
    has_input_gate: bool


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = [
            path / "adapter_latest.pt",
            path / "full_model_latest.pt",
            path / "adapter_best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No compatible checkpoint found in directory: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)}")


def infer_state_pq_rank_and_headwise(state_dict: Dict[str, torch.Tensor]) -> tuple[int, bool]:
    for key, value in state_dict.items():
        if not key.endswith("state_pq_Q"):
            continue
        if value.ndim == 3:
            return int(value.shape[1]), True
        if value.ndim == 2:
            return int(value.shape[0]), False
    for key, value in state_dict.items():
        if not key.endswith("state_pq_P"):
            continue
        if value.ndim == 3:
            return int(value.shape[2]), True
        if value.ndim == 2:
            return int(value.shape[1]), False
    return 0, False


def has_input_gate(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(
        key.endswith("input_gate_proj.weight") or key.endswith("input_gate_proj.bias")
        for key in state_dict.keys()
    )


def infer_adapter_architecture(path: str | Path) -> AdapterArchitecture:
    checkpoint_path = resolve_checkpoint_path(path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(payload)
    rank, headwise = infer_state_pq_rank_and_headwise(state_dict)
    return AdapterArchitecture(
        checkpoint_path=checkpoint_path,
        state_pq_rank=rank,
        state_pq_headwise=headwise,
        has_input_gate=has_input_gate(state_dict),
    )
