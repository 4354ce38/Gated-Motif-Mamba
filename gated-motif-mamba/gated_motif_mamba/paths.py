from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_ROOT = REPO_ROOT / "assets"
DATA_ROOT = ASSETS_ROOT / "data"
MODELS_ROOT = ASSETS_ROOT / "models"
CHECKPOINTS_ROOT = ASSETS_ROOT / "checkpoints"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

DEFAULT_BASE_MODEL = MODELS_ROOT / "mamba2-130m"
DEFAULT_DATASETS_ROOT = DATA_ROOT / "sniah"
DEFAULT_LOGS_ROOT = OUTPUTS_ROOT

DEFAULT_MAIN_ADAPTER = (
    CHECKPOINTS_ROOT
    / "130m_motif2"
    / "adapter_latest.pt"
)
DEFAULT_TRUTHFULQA_DATA_DIR = DATA_ROOT / "truthfulqa"
DEFAULT_TRUTHFULQA_LOCAL_JSONL = DEFAULT_TRUTHFULQA_DATA_DIR / "truthfulqa_mc1_validation.jsonl"
DEFAULT_TRUTHFULQA_BASE_ROWS = DEFAULT_TRUTHFULQA_DATA_DIR / "base_rows.csv"
DEFAULT_TRUTHFULQA_ADAPTER_ROWS = DEFAULT_TRUTHFULQA_DATA_DIR / "adapter_rows.csv"
DEFAULT_TRUTHFULQA_OUT_DIR = OUTPUTS_ROOT / "truthfulqa_internal_explanations"
DEFAULT_MC_CHOICE_REFUSAL_DIR = OUTPUTS_ROOT / "mc_choice_refusal"
DEFAULT_PILE_ROOT = DATA_ROOT / "pile_tiny"
DEFAULT_SNIAH_OUTPUT_DIR = OUTPUTS_ROOT / "sniah_eval_results"
