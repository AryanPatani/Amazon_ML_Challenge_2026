"""
src/path_b

Path B: Neural retrieval + cross-encoder re-ranking.
"""

from src.path_b.serialization import serialize_record, serialize_dataframe, clean_text
from src.path_b.encoder import DenseEncoder, DEFAULT_MODEL
from src.path_b.retrieval import VectorRetriever
from src.path_b.metrics import compute_recall_at_k, print_recall_report
from src.path_b.hard_negatives import mine_triplets_from_retrieval
from src.path_b.trainer import fine_tune_bi_encoder
from src.path_b.b2_pipeline import run_b2_pipeline
from src.path_b.cross_dataset import prepare_cross_encoder_examples, build_inference_pairs
from src.path_b.cross_encoder import CrossEncoderReranker, fine_tune_cross_encoder, DEFAULT_CROSS_MODEL
from src.path_b.cross_metrics import (
    find_optimal_threshold,
    apply_assignment_constraint,
    export_scores_parquet,
    export_matching_results_tsv,
)
from src.path_b.b3_pipeline import run_b3_pipeline

__all__ = [
    "serialize_record",
    "serialize_dataframe",
    "clean_text",
    "DenseEncoder",
    "DEFAULT_MODEL",
    "VectorRetriever",
    "compute_recall_at_k",
    "print_recall_report",
    "mine_triplets_from_retrieval",
    "fine_tune_bi_encoder",
    "run_b2_pipeline",
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
]


