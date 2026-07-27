__version__ = "0.1.0"

from gated_motif_mamba.checkpoint import (
    AdapterArchitecture,
    extract_state_dict,
    has_input_gate,
    infer_adapter_architecture,
    infer_state_pq_rank_and_headwise,
    resolve_checkpoint_path,
)
from gated_motif_mamba.paths import REPO_ROOT

try:
    from gated_motif_mamba.loader import build_model, greedy_generate, load_tokenizer, resolve_dtype
except ModuleNotFoundError:
    build_model = None
    greedy_generate = None
    load_tokenizer = None
    resolve_dtype = None

__all__ = [
    "__version__",
    "AdapterArchitecture",
    "REPO_ROOT",
    "build_model",
    "extract_state_dict",
    "greedy_generate",
    "has_input_gate",
    "infer_adapter_architecture",
    "infer_state_pq_rank_and_headwise",
    "load_tokenizer",
    "resolve_checkpoint_path",
    "resolve_dtype",
]
