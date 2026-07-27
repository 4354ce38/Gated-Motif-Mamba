#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gated_motif_mamba.checkpoint import infer_adapter_architecture
from gated_motif_mamba.loader import build_model, greedy_generate, load_tokenizer
from gated_motif_mamba.paths import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASETS_ROOT,
    DEFAULT_MAIN_ADAPTER,
    DEFAULT_SNIAH_OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base Mamba-2 or gated-motif-mamba on S-NIAH datasets."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_BASE_MODEL,
        help="Local Mamba-2 base model directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory. Defaults to --model-dir.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=str(DEFAULT_MAIN_ADAPTER),
        help="Optional adapter checkpoint. Use 'none' to evaluate the base model.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Root directory that contains S-NIAH dataset folders.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1024],
        help="Context lengths to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["niah_single_1", "niah_single_2", "niah_single_3"],
        help="S-NIAH tasks to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate per sample.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Torch dtype used after model loading.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of examples per task.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SNIAH_OUTPUT_DIR,
        help="Directory for predictions and summary files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def extract_answer_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    uuid_matches = re.findall(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", text)
    num_matches = re.findall(r"\b\d{7}\b", text)
    if uuid_matches:
        candidates.extend(uuid_matches)
    if num_matches:
        candidates.extend(num_matches)
    if not candidates:
        stripped = text.strip()
        if stripped:
            first_line = stripped.splitlines()[0].strip()
            if first_line:
                candidates.append(first_line)
    return candidates


def is_correct_prediction(gold_answers: Iterable[str], generated_text: str) -> Tuple[bool, List[str]]:
    normalized_generated = normalize_text(generated_text)
    candidates = [normalize_text(x) for x in extract_answer_candidates(generated_text)]
    for gold in gold_answers:
        norm_gold = normalize_text(gold)
        if norm_gold in normalized_generated:
            return True, candidates
        if any(candidate == norm_gold for candidate in candidates):
            return True, candidates
    return False, candidates


def load_jsonl(path: Path, limit: int | None) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if limit is not None and line_idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def build_prompt(sample: Dict) -> str:
    prompt = sample["input"]
    answer_prefix = sample.get("answer_prefix", "")
    if answer_prefix:
        prompt = f"{prompt}{answer_prefix}"
    return prompt


def evaluate_task(
    *,
    dataset_path: Path,
    output_path: Path,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    limit: int | None,
) -> Dict:
    rows = load_jsonl(dataset_path, limit=limit)
    if output_path.exists():
        output_path.unlink()

    correct = 0
    started = time.time()

    with output_path.open("w", encoding="utf-8") as writer:
        for sample_idx, sample in enumerate(rows, start=1):
            prompt = build_prompt(sample)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded.input_ids.to(device)
            prompt_len = int(input_ids.shape[1])

            sequences = greedy_generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_ids = sequences[:, prompt_len:]
            generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
            match, candidates = is_correct_prediction(sample["outputs"], generated_text)
            correct += int(match)

            record = {
                "index": sample.get("index", sample_idx - 1),
                "task_length": sample.get("length"),
                "gold_outputs": sample["outputs"],
                "prediction_text": generated_text,
                "prediction_candidates": candidates,
                "correct": match,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            if sample_idx % 10 == 0 or sample_idx == len(rows):
                accuracy = correct / sample_idx
                print(
                    f"[{dataset_path.parent.parent.name}/{dataset_path.parent.name}] "
                    f"{sample_idx}/{len(rows)} accuracy={accuracy:.4f}",
                    flush=True,
                )

    elapsed = time.time() - started
    return {
        "dataset": str(dataset_path),
        "prediction_file": str(output_path),
        "num_examples": len(rows),
        "num_correct": correct,
        "accuracy": (correct / len(rows)) if rows else 0.0,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.model_dir
    adapter_text = str(args.adapter).strip().lower()
    adapter = None if adapter_text in {"", "none", "null", "base"} else Path(args.adapter)

    tokenizer = load_tokenizer(tokenizer_path)
    model = build_model(
        args.model_dir,
        adapter=adapter,
        device=args.device,
        dtype=args.dtype,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict] = []
    if adapter is not None:
        arch = infer_adapter_architecture(adapter)
        print(
            f"[adapter] path={arch.checkpoint_path} "
            f"pq_rank={arch.state_pq_rank} "
            f"headwise={arch.state_pq_headwise} "
            f"input_gate={arch.has_input_gate}",
            flush=True,
        )

    for length in args.lengths:
        dataset_dir = args.datasets_root / f"S-NIAH-mamba2-130m-{length}"
        for task_name in args.tasks:
            dataset_path = dataset_dir / task_name / "test.jsonl"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            stem = f"{task_name}_{length}"
            output_path = args.output_dir / f"{stem}.jsonl"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"Prediction file already exists: {output_path}")

            result = evaluate_task(
                dataset_path=dataset_path,
                output_path=output_path,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                limit=args.limit,
            )
            result["task"] = task_name
            result["context_length"] = length
            summary.append(result)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved S-NIAH results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
