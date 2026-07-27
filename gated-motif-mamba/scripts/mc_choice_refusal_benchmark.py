#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gated_motif_mamba.paths import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MC_CHOICE_REFUSAL_DIR,
    DEFAULT_TRUTHFULQA_LOCAL_JSONL,
)


TOKENIZER_PATH = DEFAULT_BASE_MODEL
REFUSAL_TEXT = " I don't know"
DEFAULT_TASKS = [
    "truthfulqa_mc1",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "commonsenseqa",
    "sciq",
    "winogrande",
]
EXTENDED_TASKS = DEFAULT_TASKS + ["hellaswag"]


@dataclass(frozen=True)
class Example:
    question: str
    choices: List[str]
    gold_index: int
    extra: Dict[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_path: str
    dataset_config: Optional[str]
    split: str
    description: str
    expected_size: Optional[int]
    extractor: Callable[[Dict[str, Any], int], Example]


@dataclass
class EvalRow:
    task: str
    question_id: int
    category: str
    gold_letter: str
    predicted_answer: str
    gold_score: float
    refusal_score: float
    top_score: float
    question: str
    gold_choice: str


def choice_letter(index: int) -> str:
    return chr(ord("A") + index)


def get_dtype(name: str):
    import torch

    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_config(pretrained_dir: Path) -> dict:
    with (pretrained_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(base_model: Path, adapter_path: Optional[Path], device, dtype):
    from gated_motif_mamba.loader import build_model as build_gated_motif_model

    return build_gated_motif_model(
        base_model,
        adapter=adapter_path,
        device=device,
        dtype=dtype,
    )


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def stable_shuffle(items: Sequence[str], gold_index: int, seed_key: str) -> tuple[List[str], int]:
    pairs = list(enumerate(items))
    seed = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    shuffled = [text for _, text in pairs]
    new_gold = next(i for i, (old_idx, _) in enumerate(pairs) if old_idx == gold_index)
    return shuffled, new_gold


def extract_truthfulqa(row: Dict[str, Any], idx: int) -> Example:
    del idx
    question = normalize_text(row["question"])
    choices = [normalize_text(choice) for choice in row["mc1_targets"]["choices"]]
    labels = [int(value) for value in row["mc1_targets"]["labels"]]
    gold_index = labels.index(1)
    return Example(question=question, choices=choices, gold_index=gold_index, extra={})


def extract_arc_like(row: Dict[str, Any], idx: int) -> Example:
    del idx
    question = normalize_text(row["question"])
    choices = [normalize_text(choice) for choice in row["choices"]["text"]]
    labels = [str(label).strip() for label in row["choices"]["label"]]
    answer_key = str(row["answerKey"]).strip()
    gold_index = labels.index(answer_key)
    return Example(question=question, choices=choices, gold_index=gold_index, extra={})


def extract_openbookqa(row: Dict[str, Any], idx: int) -> Example:
    del idx
    question = normalize_text(row["question_stem"])
    choices = [normalize_text(choice) for choice in row["choices"]["text"]]
    labels = [str(label).strip() for label in row["choices"]["label"]]
    answer_key = str(row["answerKey"]).strip()
    gold_index = labels.index(answer_key)
    return Example(question=question, choices=choices, gold_index=gold_index, extra={})


def extract_commonsenseqa(row: Dict[str, Any], idx: int) -> Example:
    del idx
    question = normalize_text(row["question"])
    choices = [normalize_text(choice) for choice in row["choices"]["text"]]
    labels = [str(label).strip() for label in row["choices"]["label"]]
    answer_key = str(row["answerKey"]).strip()
    gold_index = labels.index(answer_key)
    return Example(question=question, choices=choices, gold_index=gold_index, extra={})


def extract_sciq(row: Dict[str, Any], idx: int) -> Example:
    question = normalize_text(row["question"])
    choices = [
        normalize_text(row["correct_answer"]),
        normalize_text(row["distractor1"]),
        normalize_text(row["distractor2"]),
        normalize_text(row["distractor3"]),
    ]
    shuffled, gold_index = stable_shuffle(choices, gold_index=0, seed_key=f"sciq::{idx}::{question}")
    return Example(question=question, choices=shuffled, gold_index=gold_index, extra={})


def extract_hellaswag(row: Dict[str, Any], idx: int) -> Example:
    del idx
    context = normalize_text(row["ctx"])
    question = f"Choose the best continuation for the following context: {context}"
    choices = [normalize_text(choice) for choice in row["endings"]]
    gold_index = int(row["label"])
    return Example(question=question, choices=choices, gold_index=gold_index, extra={})


def extract_winogrande(row: Dict[str, Any], idx: int) -> Example:
    del idx
    sentence = normalize_text(row["sentence"])
    question = f"Fill the blank in the sentence with the better option: {sentence}"
    choices = [normalize_text(row["option1"]), normalize_text(row["option2"])]
    gold_index = int(str(row["answer"]).strip()) - 1
    return Example(question=question, choices=choices, gold_index=gold_index, extra={})


TASK_SPECS: Dict[str, TaskSpec] = {
    "truthfulqa_mc1": TaskSpec(
        name="truthfulqa_mc1",
        dataset_path="truthfulqa/truthful_qa",
        dataset_config="multiple_choice",
        split="validation",
        description="Truthfulness-focused MC1 split with many plausible distractors.",
        expected_size=817,
        extractor=extract_truthfulqa,
    ),
    "arc_easy": TaskSpec(
        name="arc_easy",
        dataset_path="allenai/ai2_arc",
        dataset_config="ARC-Easy",
        split="validation",
        description="Elementary science QA, good for basic factual/commonsense abstention.",
        expected_size=570,
        extractor=extract_arc_like,
    ),
    "arc_challenge": TaskSpec(
        name="arc_challenge",
        dataset_path="allenai/ai2_arc",
        dataset_config="ARC-Challenge",
        split="validation",
        description="Harder science QA than ARC-Easy, useful for calibration under ambiguity.",
        expected_size=299,
        extractor=extract_arc_like,
    ),
    "openbookqa": TaskSpec(
        name="openbookqa",
        dataset_path="allenai/openbookqa",
        dataset_config="main",
        split="validation",
        description="Open-book science QA with short multiple-choice answers.",
        expected_size=500,
        extractor=extract_openbookqa,
    ),
    "commonsenseqa": TaskSpec(
        name="commonsenseqa",
        dataset_path="tau/commonsense_qa",
        dataset_config=None,
        split="validation",
        description="Commonsense reasoning with five-way choices.",
        expected_size=1221,
        extractor=extract_commonsenseqa,
    ),
    "sciq": TaskSpec(
        name="sciq",
        dataset_path="allenai/sciq",
        dataset_config=None,
        split="validation",
        description="Science exam questions; useful medium-sized factual benchmark.",
        expected_size=1000,
        extractor=extract_sciq,
    ),
    "winogrande": TaskSpec(
        name="winogrande",
        dataset_path="allenai/winogrande",
        dataset_config="winogrande_xl",
        split="validation",
        description="Binary pronoun/coreference disambiguation; checks fragile reasoning.",
        expected_size=1267,
        extractor=extract_winogrande,
    ),
    "hellaswag": TaskSpec(
        name="hellaswag",
        dataset_path="Rowan/hellaswag",
        dataset_config=None,
        split="validation",
        description="Large next-event / continuation benchmark; heavier but very useful.",
        expected_size=10042,
        extractor=extract_hellaswag,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice tasks with explicit refusal as a third outcome.")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--label", default="model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--tasks", default="core", help="Comma-separated task names, or one of: core, extended, all")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task row limit for smoke tests.")
    parser.add_argument("--output-dir", default=str(DEFAULT_MC_CHOICE_REFUSAL_DIR))
    parser.add_argument("--list-tasks", action="store_true")
    return parser.parse_args()


def resolve_task_names(task_arg: str) -> List[str]:
    raw = task_arg.strip().lower()
    if raw == "core":
        names = DEFAULT_TASKS
    elif raw in {"extended", "all"}:
        names = EXTENDED_TASKS
    else:
        names = [item.strip() for item in task_arg.split(",") if item.strip()]
    unknown = [name for name in names if name not in TASK_SPECS]
    if unknown:
        raise ValueError(f"Unknown task names: {', '.join(unknown)}")
    return names


def build_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [
        "Answer the following multiple-choice question.",
        "Reply with only one capital letter like A or B.",
        "Choose the single best option.",
        "If you are unsure, reply exactly: I don't know.",
        "",
        "Question: Which is larger, a mouse or an elephant?",
        "A. A mouse",
        "B. An elephant",
        "Answer: B",
        "",
        "Question: Which came first, the moon landing or the invention of the wheel?",
        "A. The moon landing",
        "B. The invention of the wheel",
        "Answer: B",
        "",
        f"Question: {question}",
    ]
    for idx, choice in enumerate(choices):
        lines.append(f"{choice_letter(idx)}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def score_candidate(model, tokenizer, prompt: str, candidate: str, device) -> float:
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    cand_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = torch.cat([prompt_ids, cand_ids], dim=1)
    with torch.no_grad():
        logits = model(full_ids).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    prompt_len = prompt_ids.shape[1]
    target = full_ids[:, 1:]
    cand_target = target[:, prompt_len - 1 :]
    cand_logprobs = logprobs[:, prompt_len - 1 :, :]
    gathered = cand_logprobs.gather(-1, cand_target.unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().item())


def load_task_dataset(spec: TaskSpec, limit: Optional[int]):
    if spec.name == "truthfulqa_mc1" and DEFAULT_TRUTHFULQA_LOCAL_JSONL.exists():
        rows = [json.loads(line) for line in DEFAULT_TRUTHFULQA_LOCAL_JSONL.read_text(encoding="utf-8").splitlines() if line]
        if limit is not None:
            rows = rows[: min(limit, len(rows))]
        return rows

    from datasets import load_dataset

    dataset = load_dataset(spec.dataset_path, spec.dataset_config, split=spec.split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def evaluate_task(
    model,
    tokenizer,
    device,
    label: str,
    output_dir: Path,
    spec: TaskSpec,
    limit: Optional[int],
) -> Dict[str, Any]:
    dataset = load_task_dataset(spec, limit)
    results: List[EvalRow] = []
    counts = {"correct": 0, "wrong": 0, "refusal": 0, "total": len(dataset)}

    for idx, raw_row in enumerate(dataset):
        example = spec.extractor(raw_row, idx)
        gold_letter = choice_letter(example.gold_index)
        gold_choice = example.choices[example.gold_index]
        prompt = build_prompt(example.question, example.choices)

        candidate_answers = {choice_letter(i): f" {choice_letter(i)}" for i in range(len(example.choices))}
        candidate_scores = {
            answer: score_candidate(model, tokenizer, prompt, answer_text, device)
            for answer, answer_text in candidate_answers.items()
        }
        refusal_score = score_candidate(model, tokenizer, prompt, REFUSAL_TEXT, device)
        candidate_scores["REFUSAL"] = refusal_score

        best_answer, top_score = max(candidate_scores.items(), key=lambda item: item[1])
        gold_score = candidate_scores[gold_letter]

        if best_answer == "REFUSAL":
            category = "refusal"
        elif best_answer == gold_letter:
            category = "correct"
        else:
            category = "wrong"
        counts[category] += 1

        results.append(
            EvalRow(
                task=spec.name,
                question_id=idx,
                category=category,
                gold_letter=gold_letter,
                predicted_answer=best_answer,
                gold_score=gold_score,
                refusal_score=refusal_score,
                top_score=top_score,
                question=example.question,
                gold_choice=gold_choice,
            )
        )

        if (idx + 1) % 50 == 0 or idx + 1 == len(dataset):
            print(f"[{label}] {spec.name}: processed {idx + 1}/{len(dataset)}", flush=True)

    rows_path = output_dir / f"{label}_{spec.name}_rows.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "question_id", "category", "gold_letter", "predicted_answer", "gold_score", "refusal_score", "top_score", "question", "gold_choice"])
        for row in results:
            writer.writerow(
                [
                    row.task,
                    row.question_id,
                    row.category,
                    row.gold_letter,
                    row.predicted_answer,
                    row.gold_score,
                    row.refusal_score,
                    row.top_score,
                    row.question,
                    row.gold_choice,
                ]
            )

    answered = counts["correct"] + counts["wrong"]
    summary = {
        "label": label,
        "task": spec.name,
        "counts": counts,
        "rates": {
            "correct": counts["correct"] / counts["total"] if counts["total"] else None,
            "wrong": counts["wrong"] / counts["total"] if counts["total"] else None,
            "refusal": counts["refusal"] / counts["total"] if counts["total"] else None,
            "answer_rate": answered / counts["total"] if counts["total"] else None,
            "answer_accuracy": counts["correct"] / answered if answered else None,
            "signed_utility": (counts["correct"] - counts["wrong"]) / counts["total"] if counts["total"] else None,
        },
        "dataset": {
            "path": spec.dataset_path,
            "config": spec.dataset_config,
            "split": spec.split,
            "description": spec.description,
        },
        "definition": {
            "correct": "Among candidate answers A/B/C/.../I don't know, the gold letter has the highest mean token logprob.",
            "wrong": "A non-gold option letter has the highest mean token logprob.",
            "refusal": "'I don't know' has the highest mean token logprob.",
        },
    }
    summary_path = output_dir / f"{label}_{spec.name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def write_aggregate_outputs(output_dir: Path, label: str, task_names: Sequence[str], summaries: Sequence[Dict[str, Any]]) -> None:
    agg_json_path = output_dir / f"{label}_task_table.json"
    with agg_json_path.open("w", encoding="utf-8") as f:
        json.dump({"label": label, "tasks": list(task_names), "summaries": list(summaries)}, f, ensure_ascii=False, indent=2)

    agg_csv_path = output_dir / f"{label}_task_table.csv"
    with agg_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "total", "correct", "wrong", "refusal", "correct_rate", "wrong_rate", "refusal_rate", "answer_rate", "answer_accuracy", "signed_utility"])
        for summary in summaries:
            counts = summary["counts"]
            rates = summary["rates"]
            writer.writerow(
                [
                    summary["task"],
                    counts["total"],
                    counts["correct"],
                    counts["wrong"],
                    counts["refusal"],
                    rates["correct"],
                    rates["wrong"],
                    rates["refusal"],
                    rates["answer_rate"],
                    rates["answer_accuracy"],
                    rates["signed_utility"],
                ]
            )

    md_lines = [
        f"# {label} refusal benchmark summary",
        "",
        "| Task | Total | Correct | Wrong | Refusal | Answer Rate | Accuracy When Answered | `(Correct - Wrong) / Total` |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        counts = summary["counts"]
        rates = summary["rates"]
        md_lines.append(
            "| "
            + " | ".join(
                [
                    summary["task"],
                    str(counts["total"]),
                    f"{counts['correct']} ({format_pct(rates['correct'])})",
                    f"{counts['wrong']} ({format_pct(rates['wrong'])})",
                    f"{counts['refusal']} ({format_pct(rates['refusal'])})",
                    format_pct(rates["answer_rate"]),
                    format_pct(rates["answer_accuracy"]),
                    format_float(rates["signed_utility"]),
                ]
            )
            + " |"
        )

    md_path = output_dir / f"{label}_task_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def print_task_catalog(task_names: Sequence[str]) -> None:
    print("| Task | Split | Expected N | Why include it |")
    print("| --- | --- | ---: | --- |")
    for name in task_names:
        spec = TASK_SPECS[name]
        size = spec.expected_size if spec.expected_size is not None else "?"
        print(f"| {spec.name} | {spec.split} | {size} | {spec.description} |")


def main() -> None:
    args = parse_args()
    task_names = resolve_task_names(args.tasks)

    if args.list_tasks:
        print_task_catalog(task_names)
        return

    base_model = Path(args.base_model)
    adapter = Path(args.adapter) if args.adapter else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer

    device_name = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    dtype = get_dtype(args.dtype)

    print(f"[load] base_model={base_model}")
    print(f"[load] adapter={adapter}")
    print(f"[load] device={device} dtype={dtype}")
    print(f"[tasks] {', '.join(task_names)}")

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    model = build_model(base_model, adapter, device, dtype)

    summaries = []
    for task_name in task_names:
        spec = TASK_SPECS[task_name]
        print(f"[task] starting {task_name}", flush=True)
        summaries.append(
            evaluate_task(
                model=model,
                tokenizer=tokenizer,
                device=device,
                label=args.label,
                output_dir=output_dir,
                spec=spec,
                limit=args.limit,
            )
        )

    write_aggregate_outputs(output_dir, args.label, task_names, summaries)
    print(json.dumps({"label": args.label, "tasks": task_names, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
