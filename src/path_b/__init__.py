"""
src/path_b

Path B: Neural retrieval + cross-encoder re-ranking.
Tickets B1 through B5.
"""

# B1: Serialization, Dense Encoder, Retrieval, Recall Metrics
from src.path_b.serialization import serialize_record, serialize_dataframe, clean_text
from src.path_b.encoder import DenseEncoder, DEFAULT_MODEL
from src.path_b.retrieval import VectorRetriever
from src.path_b.metrics import compute_recall_at_k, print_recall_report

# B2: Hard Negative Mining, Contrastive Fine-Tuning
from src.path_b.hard_negatives import mine_triplets_from_retrieval, InputExample
from src.path_b.trainer import fine_tune_bi_encoder

# B3: Cross-Encoder Dataset, Re-Ranking, Thresholding, 1-to-1 Constraint
from src.path_b.cross_dataset import prepare_cross_encoder_examples, build_inference_pairs
from src.path_b.cross_encoder import CrossEncoderReranker, fine_tune_cross_encoder, DEFAULT_CROSS_MODEL
from src.path_b.cross_metrics import (
    find_optimal_threshold,
    apply_assignment_constraint,
    export_scores_parquet,
    export_matching_results_tsv,
)

# B4: Synthetic Noise Augmentation
from src.path_b.augmentation import corrupt_text, augment_batch

# B5: Neural Feature Export for Path A & D Ensembles
from src.path_b.feature_export import (
    build_path_b_feature_dataframe,
    export_path_b_parquet,
    load_path_b_features,
    validate_path_b_dataframe,
)

# Lazy pipeline loaders (PEP 562) to prevent runpy RuntimeWarnings when executing via python -m
_LAZY_PIPELINES = {
    "run_b1_retrieval": ("src.path_b.b1_pipeline", "run_b1_retrieval"),
    "run_b1_pipeline":  ("src.path_b.b1_pipeline", "run_b1_retrieval"),  # alias
    "run_b2_pipeline":  ("src.path_b.b2_pipeline", "run_b2_pipeline"),
    "run_b3_pipeline":  ("src.path_b.b3_pipeline", "run_b3_pipeline"),
    "run_b4_pipeline":  ("src.path_b.b4_pipeline", "run_b4_pipeline"),
    "run_b5_export":    ("src.path_b.b5_pipeline", "run_b5_export"),
    "run_b5_pipeline":  ("src.path_b.b5_pipeline", "run_b5_export"),  # alias
}


def __getattr__(name: str):
    if name in _LAZY_PIPELINES:
        mod_name, func_name = _LAZY_PIPELINES[name]
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, func_name)
    raise AttributeError(f"module 'src.path_b' has no attribute '{name}'")


__all__ = [
    # B1
    "serialize_record",
    "serialize_dataframe",
    "clean_text",
    "DenseEncoder",
    "DEFAULT_MODEL",
    "VectorRetriever",
    "compute_recall_at_k",
    "print_recall_report",
    "run_b1_retrieval",
    "run_b1_pipeline",
    # B2
    "mine_triplets_from_retrieval",
    "InputExample",
    "fine_tune_bi_encoder",
    "run_b2_pipeline",
    # B3
    "prepare_cross_encoder_examples",
    "build_inference_pairs",
    "CrossEncoderReranker",
    "fine_tune_cross_encoder",
    "DEFAULT_CROSS_MODEL",
    "find_optimal_threshold",
    "apply_assignment_constraint",
    "export_scores_parquet",
    "export_matching_results_tsv",
    "run_b3_pipeline",
    # B4
    "corrupt_text",
    "augment_batch",
    "run_b4_pipeline",
    # B5
    "build_path_b_feature_dataframe",
    "export_path_b_parquet",
    "load_path_b_features",
    "validate_path_b_dataframe",
    "run_b5_export",
    "run_b5_pipeline",
]
