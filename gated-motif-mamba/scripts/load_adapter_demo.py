#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gated_motif_mamba.checkpoint import infer_adapter_architecture
from gated_motif_mamba.loader import build_model, greedy_generate, load_tokenizer
from gated_motif_mamba.paths import DEFAULT_BASE_MODEL, DEFAULT_MAIN_ADAPTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal loader demo for gated-motif-mamba.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_MAIN_ADAPTER,
    )
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--prompt", type=str, default="Q: What is the capital of France?\nA:")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path or args.model_dir)
    arch = infer_adapter_architecture(args.adapter)
    print(
        f"[adapter] path={arch.checkpoint_path}\n"
        f"[adapter] pq_rank={arch.state_pq_rank} headwise={arch.state_pq_headwise} input_gate={arch.has_input_gate}",
        flush=True,
    )

    model = build_model(
        args.model_dir,
        adapter=args.adapter,
        device=args.device,
        dtype=args.dtype,
    )
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(args.device)
    sequences = greedy_generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    print("[generation]")
    print(generated_text)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
