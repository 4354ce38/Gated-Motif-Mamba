from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from gated_motif_mamba.checkpoint import (
    extract_state_dict,
    infer_adapter_architecture,
    resolve_checkpoint_path,
)
from gated_motif_mamba_ssm.models.config_mamba import MambaConfig
from gated_motif_mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {None, "auto"}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(dtype)]


def load_tokenizer(tokenizer_path: str | Path):
    return AutoTokenizer.from_pretrained(str(Path(tokenizer_path).expanduser().resolve()))


def _load_config(base_model: Path) -> MambaConfig:
    with (base_model / "config.json").open("r", encoding="utf-8") as handle:
        return MambaConfig(**json.load(handle))


def build_model(
    base_model: str | Path,
    *,
    adapter: str | Path | None = None,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype | None = "float16",
    chunk_size: int = 64,
    input_gate_mode: str = "symmetric_exp",
    input_gate_gain: float = 0.5,
    input_gate_chunk_reduce: str = "importance",
    input_gate_chunk_temp: float = 4.0,
    state_pq_alpha: float = 1e-2,
    state_pq_apply_to: str = "running",
) -> MambaLMHeadModel:
    base_model = Path(base_model).expanduser().resolve()
    device_obj = torch.device(device)
    dtype_obj = resolve_dtype(dtype)

    adapter_state = None
    adapter_arch = None
    if adapter:
        adapter_path = resolve_checkpoint_path(adapter)
        adapter_state = extract_state_dict(torch.load(adapter_path, map_location="cpu"))
        adapter_arch = infer_adapter_architecture(adapter_path)

    config = _load_config(base_model)
    ssm_cfg = dict(config.ssm_cfg or {})
    ssm_cfg["layer"] = "Mamba2"
    ssm_cfg["use_mem_eff_path"] = False
    ssm_cfg["chunk_size"] = int(chunk_size)
    ssm_cfg["input_gate_on_state"] = bool(adapter_arch.has_input_gate) if adapter_arch else False
    ssm_cfg["input_gate_mode"] = str(input_gate_mode)
    ssm_cfg["input_gate_gain"] = float(input_gate_gain)
    ssm_cfg["input_gate_chunk_reduce"] = str(input_gate_chunk_reduce)
    ssm_cfg["input_gate_chunk_temp"] = float(input_gate_chunk_temp)
    ssm_cfg["state_pq_rank"] = int(adapter_arch.state_pq_rank) if adapter_arch else 0
    ssm_cfg["state_pq_alpha"] = float(state_pq_alpha)
    ssm_cfg["state_pq_apply_to"] = str(state_pq_apply_to)
    ssm_cfg["state_pq_headwise"] = bool(adapter_arch.state_pq_headwise) if adapter_arch else False
    config.ssm_cfg = ssm_cfg

    model = MambaLMHeadModel(config, device=device_obj, dtype=dtype_obj)
    base_state = torch.load(base_model / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(base_state, strict=False)
    if adapter_state is not None:
        model.load_state_dict(adapter_state, strict=False)
    model = model.to(device_obj)
    model.eval()
    return model


def greedy_generate(
    model: MambaLMHeadModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    sequences = input_ids.clone()
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0
    finished = torch.zeros(sequences.shape[0], dtype=torch.bool, device=sequences.device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(
                input_ids=sequences,
                inference_params=None,
                num_last_tokens=1,
            ).logits[:, -1, :]
            next_tokens = logits.argmax(dim=-1, keepdim=True)
            if finished.any():
                next_tokens = next_tokens.masked_fill(finished.unsqueeze(1), pad_token_id)
            sequences = torch.cat([sequences, next_tokens], dim=1)
            if eos_token_id is not None:
                finished |= next_tokens.squeeze(1).eq(eos_token_id)
            if bool(finished.all()):
                break
    return sequences
