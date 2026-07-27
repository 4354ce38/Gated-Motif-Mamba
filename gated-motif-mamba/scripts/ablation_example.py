#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gated_motif_mamba.paths import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASETS_ROOT,
    DEFAULT_GATE_ONLY_ADAPTER,
    DEFAULT_MAIN_ADAPTER,
    DEFAULT_PQ_ONLY_ADAPTER,
)

BASE_MODEL = DEFAULT_BASE_MODEL


@dataclass(frozen=True)
class AblationRecipe:
    name: str
    adapter: Path | None
    note: str


RECIPES = [
    AblationRecipe("base", None, "No adapter, plain Mamba-2 baseline."),
    AblationRecipe(
        "gate_only",
        DEFAULT_GATE_ONLY_ADAPTER,
        "Only input gate enabled.",
    ),
    AblationRecipe(
        "pq_only",
        DEFAULT_PQ_ONLY_ADAPTER,
        "Only state PQ enabled.",
    ),
    AblationRecipe(
        "motif2_rank2",
        DEFAULT_MAIN_ADAPTER,
        "Full gated motif adapter used in the paper figures.",
    ),
]


def main() -> None:
    print("# Example ablation commands")
    for recipe in RECIPES:
        adapter_flag = "" if recipe.adapter is None else f" --adapter {recipe.adapter}"
        exists = recipe.adapter is None or recipe.adapter.exists()
        status = "exists" if exists else "missing"
        print(f"\n[{recipe.name}] {recipe.note} ({status})")
        print(
            "python scripts/run_sniah_mamba2_eval.py "
            f"--model-dir {BASE_MODEL}{adapter_flag} "
            f"--output-dir {DEFAULT_DATASETS_ROOT / 'sniah_eval_results' / recipe.name}"
        )


if __name__ == "__main__":
    main()
