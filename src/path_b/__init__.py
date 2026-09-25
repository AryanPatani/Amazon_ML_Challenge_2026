"""
src/path_b

Path B: Neural retrieval + cross-encoder re-ranking.
"""

from src.path_b.serialization import serialize_record, serialize_dataframe, clean_text
from src.path_b.encoder import DenseEncoder, DEFAULT_MODEL
from src.path_b.retrieval import VectorRetriever
from src.path_b.metrics import compute_recall_at_k, print_recall_report

__all__ = [
    "serialize_record",
    "serialize_dataframe",
    "clean_text",
    "DenseEncoder",
    "DEFAULT_MODEL",
    "VectorRetriever",
    "compute_recall_at_k",
    "print_recall_report",
]
