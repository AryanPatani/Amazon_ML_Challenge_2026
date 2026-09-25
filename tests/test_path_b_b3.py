"""
tests/test_path_b_b3.py

Unit tests for Path B Ticket B3:
- Cross-Encoder training pair generation
- Inference pair flattening
- Global 1-to-1 assignment constraint
- Threshold sweep and Macro F0.5 optimization
- Shared parquet score file contract (s1_id, cand_id, score)
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_b.cross_dataset import prepare_cross_encoder_examples, build_inference_pairs
from src.path_b.cross_metrics import (
    apply_assignment_constraint,
    find_optimal_threshold,
    format_scored_pairs_dataframe,
    export_scores_parquet,
    export_matching_results_tsv,
)


def test_prepare_cross_encoder_examples():
    s1_ids = ["S1-1", "S1-2"]
    s1_texts = {
        "S1-1": "Acme Corp | 100 Main St",
        "S1-2": "Beta LLC | 200 Elm St",
    }
    corpus_texts = {
        "S2-1": "Acme Corporation | 100 Main St",  # Match for S1-1
        "S2-2": "Acme Tools | 100 Main St",        # Confuser for S1-1
        "S2-3": "Beta Limited | 200 Elm St",       # Match for S1-2
        "S2-4": "Beta Consulting | 999 Oak St",   # Confuser for S1-2
        "S2-5": "Gamma Global | 555 Pine St",      # Unrelated
    }
    ground_truth = {
        "S1-1": ["S2-1"],
        "S1-2": ["S2-3"],
    }
    retrieval_candidates = {
        "S1-1": ["S2-1", "S2-2"],
        "S1-2": ["S2-3", "S2-4"],
    }

    # When include_fallback_negatives=False, only retrieved confusers are used
    examples_no_fallback = prepare_cross_encoder_examples(
        s1_ids=s1_ids,
        s1_texts=s1_texts,
        corpus_texts=corpus_texts,
        ground_truth=ground_truth,
        retrieval_candidates=retrieval_candidates,
        max_negatives_per_positive=2,
        include_fallback_negatives=False,
    )

    # 2 positive pairs (S1-1 -> S2-1, S1-2 -> S2-3)
    # 2 retrieved hard negative pairs (S1-1 -> S2-2, S1-2 -> S2-4)
    assert len(examples_no_fallback) == 4
    labels_no_fb = [ex.label for ex in examples_no_fallback]
    assert labels_no_fb.count(1.0) == 2
    assert labels_no_fb.count(0.0) == 2

    # When include_fallback_negatives=True, it pads up to max_negatives_per_positive (target_negs=2 each)
    examples_with_fallback = prepare_cross_encoder_examples(
        s1_ids=s1_ids,
        s1_texts=s1_texts,
        corpus_texts=corpus_texts,
        ground_truth=ground_truth,
        retrieval_candidates=retrieval_candidates,
        max_negatives_per_positive=2,
        include_fallback_negatives=True,
    )
    assert len(examples_with_fallback) == 6
    labels_fb = [ex.label for ex in examples_with_fallback]
    assert labels_fb.count(1.0) == 2
    assert labels_fb.count(0.0) == 4

    # Check content of first positive
    pos_ex = [ex for ex in examples_no_fallback if ex.label == 1.0 and ex.texts[0] == s1_texts["S1-1"]][0]
    assert pos_ex.texts[1] == corpus_texts["S2-1"]

    # Check content of first negative
    neg_ex = [ex for ex in examples_no_fallback if ex.label == 0.0 and ex.texts[0] == s1_texts["S1-1"]][0]
    assert neg_ex.texts[1] == corpus_texts["S2-2"]



def test_build_inference_pairs():
    candidate_dict = {
        "S1-1": [("S2-1", 0.95), ("S2-2", 0.80)],
        "S1-2": ["S2-3"],
    }
    s1_texts = {
        "S1-1": "Text S1-1",
        "S1-2": "Text S1-2",
    }
    corpus_texts = {
        "S2-1": "Text S2-1",
        "S2-2": "Text S2-2",
        "S2-3": "Text S2-3",
    }

    text_pairs, id_pairs = build_inference_pairs(candidate_dict, s1_texts, corpus_texts)

    assert len(text_pairs) == 3
    assert len(id_pairs) == 3
    assert id_pairs[0] == ("S1-1", "S2-1")
    assert text_pairs[0] == ("Text S1-1", "Text S2-1")
    assert id_pairs[1] == ("S1-1", "S2-2")
    assert id_pairs[2] == ("S1-2", "S2-3")


def test_apply_assignment_constraint():
    # Candidate S2-CONFLICT appears under both S1-A (score 0.85) and S1-B (score 0.70)
    predictions = {
        "S1-A": [("S2-A", 0.90), ("S2-CONFLICT", 0.85)],
        "S1-B": [("S2-CONFLICT", 0.70), ("S2-B", 0.80)],
    }

    # At threshold 0.60, S2-CONFLICT should go to S1-A only
    assigned = apply_assignment_constraint(predictions, threshold=0.60)

    assert "S2-CONFLICT" in assigned["S1-A"]
    assert "S2-CONFLICT" not in assigned["S1-B"]
    assert "S2-A" in assigned["S1-A"]
    assert "S2-B" in assigned["S1-B"]


def test_find_optimal_threshold():
    scored_candidates = {
        "S1-1": [("S2-1", 0.92), ("S2-2", 0.35)],  # S2-1 is true match
        "S1-2": [("S2-3", 0.88), ("S2-4", 0.15)],  # S2-3 is true match
        "S1-3": [("S2-5", 0.20)],                  # Singleton (true match is empty)
    }
    ground_truth = {
        "S1-1": ["S2-1"],
        "S1-2": ["S2-3"],
        "S1-3": [],  # Singleton
    }

    result = find_optimal_threshold(
        scored_candidates=scored_candidates,
        ground_truth=ground_truth,
        threshold_range=[0.10, 0.30, 0.50, 0.70, 0.90],
        enforce_one_to_one=True,
    )

    assert "best_threshold" in result
    assert "best_macro_f05" in result
    assert "auc_metrics" in result
    assert result["auc_metrics"]["roc_auc"] > 0.90

    # At threshold 0.50 or 0.70, predictions should be perfect (F0.5 = 1.0)
    assert result["best_macro_f05"] == 1.0
    best_preds = result["best_predictions"]
    assert best_preds["S1-1"] == ["S2-1"]
    assert best_preds["S1-2"] == ["S2-3"]
    assert best_preds["S1-3"] == []  # Singleton correctly returns empty list


def test_apply_assignment_constraint_with_min_top_score():
    predictions = {
        "S1-STRONG": [("S2-1", 0.85), ("S3-1", 0.65)],
        "S1-BORDERLINE": [("S2-2", 0.60)],  # Max score is 0.60
    }

    # At threshold 0.50, without min_top_score, both entities produce matches
    assigned_no_trigger = apply_assignment_constraint(predictions, threshold=0.50)
    assert assigned_no_trigger["S1-STRONG"] == ["S2-1", "S3-1"]
    assert assigned_no_trigger["S1-BORDERLINE"] == ["S2-2"]

    # At threshold 0.50, with min_top_score=0.70:
    # S1-STRONG triggers because max score 0.85 >= 0.70, and collects both S2-1 (0.85) and S3-1 (0.65)
    # S1-BORDERLINE fails trigger because max score 0.60 < 0.70, so it abstains completely ([])
    assigned_trigger = apply_assignment_constraint(predictions, threshold=0.50, min_top_score=0.70)
    assert assigned_trigger["S1-STRONG"] == ["S2-1", "S3-1"]
    assert assigned_trigger["S1-BORDERLINE"] == []


def test_scores_parquet_contract():
    scored_candidates = {
        "S1-1": [("S2-1", 0.95), ("S2-2", 0.12)],
        "S1-2": [("S2-3", 0.88)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test_path_b_val.parquet"
        export_scores_parquet(scored_candidates, parquet_path)

        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)

        # Team contract verification
        assert list(df.columns) == ["s1_id", "cand_id", "score"]
        assert len(df) == 3
        assert df["score"].min() >= 0.0
        assert df["score"].max() <= 1.0
        assert df["s1_id"].dtype == object or str(df["s1_id"].dtype) == "string"


def test_export_matching_results_tsv():
    preds = {
        "S1-1": ["S2-1", "S3-1"],
        "S1-2": ["S2-2"],
        "S1-3": [],  # Singleton
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "matching_results.tsv"
        export_matching_results_tsv(preds, tsv_path)

        assert tsv_path.exists()
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)

        assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"]
        assert len(df) == 3
        assert df.loc[df["source1_entity_id"] == "S1-1", "matched_entity_ids"].values[0] == "S2-1,S3-1"
        assert df.loc[df["source1_entity_id"] == "S1-3", "matched_entity_ids"].values[0] == ""


def test_candidate_pairs_tsv_roundtrip():
    from src.path_b.retrieval import VectorRetriever
    candidates = {
        "S1-1": [("S2-1", 0.95), ("S2-2", 0.80)],
        "S1-2": ["S3-1", "S3-2"],
        "S1-3": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cands_path = Path(tmpdir) / "candidates.tsv"
        VectorRetriever.export_candidate_pairs_tsv(candidates, cands_path)
        assert cands_path.exists()

        loaded = VectorRetriever.load_candidate_pairs_tsv(cands_path)
        assert loaded["S1-1"] == ["S2-1", "S2-2"]
        assert loaded["S1-2"] == ["S3-1", "S3-2"]
        assert loaded["S1-3"] == []


if __name__ == "__main__":
    test_prepare_cross_encoder_examples()
    test_build_inference_pairs()
    test_apply_assignment_constraint()
    test_apply_assignment_constraint_with_min_top_score()
    test_find_optimal_threshold()
    test_scores_parquet_contract()
    test_export_matching_results_tsv()
    test_candidate_pairs_tsv_roundtrip()
    print("All Ticket B3 unit tests passed successfully!")
