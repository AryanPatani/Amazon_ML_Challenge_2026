"""
tests/test_path_b.py

Unit tests for Path B Ticket B1:
- Serialization logic
- Vector retrieval ranking
- Recall@K metric calculations
"""

import math
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.path_b.serialization import serialize_record, serialize_dataframe, clean_text
from src.path_b.retrieval import VectorRetriever
from src.path_b.metrics import compute_recall_at_k


def test_serialization():
    # Basic
    res = serialize_record("Acme Corp", "123 Main St", "US", drop_country=True)
    assert res == "Acme Corp | 123 Main St"

    # With country
    res_country = serialize_record("Acme Corp", "123 Main St", "US", drop_country=False)
    assert res_country == "Acme Corp | 123 Main St | US"

    # With prefixes
    res_pref = serialize_record("Acme Corp", "123 Main St", "US", drop_country=True, prefix_fields=True)
    assert res_pref == "name: Acme Corp | address: 123 Main St"

    # Missing address
    res_no_addr = serialize_record("Acme Corp", "", drop_country=True)
    assert res_no_addr == "Acme Corp"

    # DataFrame serialization
    df = pd.DataFrame([
        {"business_name": "ABC Ltd", "business_address": "456 Elm St", "country": "India"},
        {"business_name": "XYZ Inc", "business_address": "", "country": "US"},
    ])
    lines = serialize_dataframe(df, drop_country=True)
    assert len(lines) == 2
    assert lines[0] == "ABC Ltd | 456 Elm St"
    assert lines[1] == "XYZ Inc"


def test_vector_retrieval_ranking():
    # 3 corpus items with dim=4 (normalized)
    corpus_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    corpus_ids = ["C1", "C2", "C3"]

    retriever = VectorRetriever(corpus_vecs, corpus_ids, use_faiss=False, device="cpu")

    # Query close to C2
    query_vecs = np.array([
        [0.1, 0.99, 0.0, 0.0],
    ], dtype=np.float32)
    # L2 normalize
    query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_ids = ["Q1"]

    results = retriever.search_top_k(query_vecs, query_ids, top_k=2, show_progress_bar=False)

    assert "Q1" in results
    assert len(results["Q1"]) == 2
    top1_id, top1_score = results["Q1"][0]
    assert top1_id == "C2"
    assert top1_score > 0.95


def test_recall_metric_calculation():
    # Mock ground truth:
    # Q1 -> [C1, C2]
    # Q2 -> [C3]
    # Q3 -> [] (singleton)
    ground_truth = {
        "Q1": ["C1", "C2"],
        "Q2": ["C3"],
        "Q3": [],
    }

    # Mock retrieval:
    # Q1: [C1, X, Y, C2] -> C1 is at rank 1, C2 is at rank 4
    # Q2: [X, C3]        -> C3 is at rank 2
    # Q3: [X, Y]         -> singleton (ignored for recall)
    retrieved = {
        "Q1": ["C1", "X", "Y", "C2"],
        "Q2": ["X", "C3"],
        "Q3": ["X", "Y"],
    }

    metrics = compute_recall_at_k(retrieved, ground_truth, k_list=[1, 2, 4])

    assert metrics["num_queries"] == 3
    assert metrics["num_non_singletons"] == 2
    assert metrics["num_singletons"] == 1
    assert metrics["total_true_matches"] == 3

    # At K=1:
    # Q1 hits {C1} (1/2), Q2 hits {} (0/1). Micro: 1/3, Macro: (0.5 + 0.0)/2 = 0.25
    assert math.isclose(metrics["micro_recall_at_k"][1], 1 / 3, rel_tol=1e-2)
    assert math.isclose(metrics["macro_recall_at_k"][1], 0.25, rel_tol=1e-2)

    # At K=2:
    # Q1 hits {C1} (1/2), Q2 hits {C3} (1/1). Micro: 2/3, Macro: (0.5 + 1.0)/2 = 0.75
    assert math.isclose(metrics["micro_recall_at_k"][2], 2 / 3, rel_tol=1e-2)
    assert math.isclose(metrics["macro_recall_at_k"][2], 0.75, rel_tol=1e-2)

    # At K=4:
    # Both queries hit 100% of their matches. Micro: 3/3 = 1.0, Macro = 1.0
    assert math.isclose(metrics["micro_recall_at_k"][4], 1.0, rel_tol=1e-2)
    assert math.isclose(metrics["macro_recall_at_k"][4], 1.0, rel_tol=1e-2)
    assert math.isclose(metrics["all_hits_at_k"][4], 1.0, rel_tol=1e-2)


def test_canonical_serialization():
    # Canonical suffix expansion
    s1 = serialize_record("Modern Infra Private Limited", "123 Main Rd", canonicalize=True)
    s2 = serialize_record("Modern Infra Pvt. Ltd.", "123 Main Road", canonicalize=True)
    assert s1 == s2, f"Expected canonical equality, got: '{s1}' vs '{s2}'"


def test_country_partitioned_retrieval():
    # 2 US corpus items, 1 India corpus item
    corpus_vecs = np.array([
        [1.0, 0.0, 0.0],  # US item
        [0.9, 0.1, 0.0],  # India item (closer vector!)
        [0.0, 1.0, 0.0],  # US item
    ], dtype=np.float32)
    corpus_ids = ["US-1", "IN-1", "US-2"]
    corpus_countries = ["US", "India", "US"]

    retriever = VectorRetriever(
        corpus_embeddings=corpus_vecs,
        corpus_ids=corpus_ids,
        corpus_countries=corpus_countries,
        use_faiss=False,
        device="cpu",
    )

    # Query is US, closest vector is technically IN-1 (index 1), but country is US
    query_vecs = np.array([[0.95, 0.05, 0.0]], dtype=np.float32)
    query_ids = ["Q-US"]
    query_countries = ["US"]

    results = retriever.search_top_k(
        query_embeddings=query_vecs,
        query_ids=query_ids,
        query_countries=query_countries,
        top_k=2,
        partition_by_country=True,
        show_progress_bar=False,
    )

    retrieved_ids = [cand_id for cand_id, _ in results["Q-US"]]
    # IN-1 must NOT be in the results!
    assert "IN-1" not in retrieved_ids
    assert "US-1" in retrieved_ids


def test_candidate_tsv_export(tmp_path=None):
    from pathlib import Path
    import tempfile

    retrieval_results = {
        "S1-1": [("S2-10", 0.95), ("S3-20", 0.88)],
        "S1-2": [("S2-30", 0.72)],
    }
    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "candidate_pairs.tsv"
        VectorRetriever.export_candidate_pairs_tsv(retrieval_results, out_file)
        assert out_file.exists()

        df = pd.read_csv(out_file, sep="\t")
        assert list(df.columns) == ["source1_entity_id", "candidate_entity_ids"]
        assert len(df) == 2
        assert df.loc[df["source1_entity_id"] == "S1-1", "candidate_entity_ids"].values[0] == "S2-10,S3-20"


if __name__ == "__main__":
    test_serialization()
    test_canonical_serialization()
    test_vector_retrieval_ranking()
    test_country_partitioned_retrieval()
    test_recall_metric_calculation()
    test_candidate_tsv_export()
    print("All Path B unit tests passed successfully!")
