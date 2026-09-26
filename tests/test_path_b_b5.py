"""
tests/test_path_b_b5.py

Ticket B5 unit tests: Feature export for downstream models (Path A and Path D).
Tests:
- Schema compliance with the shared team contract: s1_id, cand_id, score in [0, 1]
- Extra features: bi_encoder_score, retrieval_rank, retrieval_rr, score_margin, bi_encoder_margin
- Ground-truth target column 'is_match' for validation split
- Validation checks for nulls, out-of-range values, and type mismatch
- Parquet export and roundtrip loading
- Integration and merge compatibility with Path A feature tables
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_b.feature_export import (
    build_path_b_feature_dataframe,
    export_path_b_parquet,
    load_path_b_features,
    validate_path_b_dataframe,
)
from src.path_b.cross_metrics import export_scores_parquet
from src.path_b.b5_pipeline import run_demo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_candidate_data():
    cross_scores = {
        "S1-1001": [("S2-2001", 0.95), ("S3-3001", 0.40), ("S2-2002", 0.15)],
        "S1-1002": [("S3-3002", 0.85), ("S2-2003", 0.30)],
        "S1-1003": [("S2-2004", 0.10)],  # Singleton with weak candidate
    }
    bi_candidates = {
        "S1-1001": [("S2-2001", 0.88), ("S3-3001", 0.75), ("S2-2002", 0.62)],
        "S1-1002": [("S3-3002", 0.82), ("S2-2003", 0.65)],
        "S1-1003": [("S2-2004", 0.55)],
    }
    gt = {
        "S1-1001": ["S2-2001"],
        "S1-1002": ["S3-3002"],
        "S1-1003": [],  # Singleton (no true match)
    }
    return cross_scores, bi_candidates, gt


# ---------------------------------------------------------------------------
# Feature DataFrame Construction Tests
# ---------------------------------------------------------------------------

class TestFeatureDataframeBuilder:
    def test_required_contract_columns_present(self, mock_candidate_data):
        cross_scores, bi_cands, gt = mock_candidate_data
        df = build_path_b_feature_dataframe(
            cross_scored_candidates=cross_scores,
            bi_encoder_candidates=bi_cands,
            ground_truth=gt,
        )
        assert "s1_id" in df.columns
        assert "cand_id" in df.columns
        assert "score" in df.columns

    def test_extra_neural_features_present(self, mock_candidate_data):
        cross_scores, bi_cands, gt = mock_candidate_data
        df = build_path_b_feature_dataframe(
            cross_scored_candidates=cross_scores,
            bi_encoder_candidates=bi_cands,
            ground_truth=gt,
        )
        expected_features = [
            "cross_encoder_score",
            "bi_encoder_score",
            "retrieval_rank",
            "retrieval_rr",
            "score_margin",
            "bi_encoder_margin",
            "score_ratio",
            "bi_encoder_ratio",
            "rank_discounted_score",
            "is_match",
        ]
        for feat in expected_features:
            assert feat in df.columns, f"Expected feature {feat} missing from DataFrame"

    def test_score_range_bounded_0_to_1(self):
        # Raw scores outside [0, 1] should be clipped safely
        cross_scores = {"S1-1": [("S2-1", 1.25), ("S2-2", -0.15)]}
        df = build_path_b_feature_dataframe(cross_scored_candidates=cross_scores)
        assert (df["score"] >= 0.0).all()
        assert (df["score"] <= 1.0).all()
        assert df.loc[df["cand_id"] == "S2-1", "score"].values[0] == 1.0
        assert df.loc[df["cand_id"] == "S2-2", "score"].values[0] == 0.0

    def test_retrieval_rank_1_indexed(self, mock_candidate_data):
        cross_scores, bi_cands, _ = mock_candidate_data
        df = build_path_b_feature_dataframe(
            cross_scored_candidates=cross_scores,
            bi_encoder_candidates=bi_cands,
        )
        s1_rows = df[df["s1_id"] == "S1-1001"].sort_values("retrieval_rank")
        ranks = s1_rows["retrieval_rank"].tolist()
        assert ranks == [1, 2, 3]

    def test_retrieval_rr_calculation(self, mock_candidate_data):
        cross_scores, bi_cands, _ = mock_candidate_data
        df = build_path_b_feature_dataframe(
            cross_scored_candidates=cross_scores,
            bi_encoder_candidates=bi_cands,
        )
        for _, row in df.iterrows():
            expected_rr = 1.0 / float(row["retrieval_rank"])
            assert abs(row["retrieval_rr"] - expected_rr) < 1e-6

    def test_score_margin_calculation(self, mock_candidate_data):
        cross_scores, bi_cands, _ = mock_candidate_data
        df = build_path_b_feature_dataframe(
            cross_scored_candidates=cross_scores,
            bi_encoder_candidates=bi_cands,
        )
        s1_rows = df[df["s1_id"] == "S1-1001"]
        top_cand = s1_rows[s1_rows["cand_id"] == "S2-2001"].iloc[0]
        # Top candidate margin should be 0.0
        assert abs(top_cand["score_margin"]) < 1e-6
        # Other candidate margin should be negative
        second_cand = s1_rows[s1_rows["cand_id"] == "S3-3001"].iloc[0]
        assert second_cand["score_margin"] < 0.0
        assert abs(second_cand["score_margin"] - (0.40 - 0.95)) < 1e-5

    def test_ground_truth_labeling(self, mock_candidate_data):
        cross_scores, bi_cands, gt = mock_candidate_data
        df = build_path_b_feature_dataframe(
            cross_scored_candidates=cross_scores,
            bi_encoder_candidates=bi_cands,
            ground_truth=gt,
        )
        # S1-1001 matches S2-2001
        m1 = df[(df["s1_id"] == "S1-1001") & (df["cand_id"] == "S2-2001")]["is_match"].values[0]
        assert m1 == 1
        # S1-1001 does not match S3-3001
        m2 = df[(df["s1_id"] == "S1-1001") & (df["cand_id"] == "S3-3001")]["is_match"].values[0]
        assert m2 == 0
        # S1-1003 is singleton -> all candidates are 0
        m3 = df[df["s1_id"] == "S1-1003"]["is_match"].values[0]
        assert m3 == 0

    def test_unretrieved_candidate_fallback_rank(self):
        # Candidate was evaluated by cross-encoder but was not in bi-encoder top-k
        cross_scores = {"S1-1": [("S2-1", 0.90), ("S2-Unseen", 0.50)]}
        bi_cands = {"S1-1": [("S2-1", 0.85)]}
        df = build_path_b_feature_dataframe(
            cross_scored_candidates=cross_scores,
            bi_encoder_candidates=bi_cands,
        )
        unseen_row = df[df["cand_id"] == "S2-Unseen"].iloc[0]
        assert unseen_row["retrieval_rank"] == 2  # max_rank (1) + 1
        assert unseen_row["bi_encoder_score"] == 0.0

    def test_empty_candidate_dict_returns_empty_dataframe(self):
        df = build_path_b_feature_dataframe(cross_scored_candidates={})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Contract Validation Tests
# ---------------------------------------------------------------------------

class TestParquetContractValidation:
    def test_valid_dataframe_passes(self, mock_candidate_data):
        cross_scores, bi_cands, _ = mock_candidate_data
        df = build_path_b_feature_dataframe(cross_scores, bi_cands)
        is_valid, errors = validate_path_b_dataframe(df)
        assert is_valid, f"Validation failed with: {errors}"
        assert len(errors) == 0

    def test_missing_required_column_fails(self):
        df = pd.DataFrame({"s1_id": ["S1-1"], "cand_id": ["S2-1"]})  # missing 'score'
        is_valid, errors = validate_path_b_dataframe(df)
        assert not is_valid
        assert any("score" in err for err in errors)

    def test_null_value_fails(self):
        df = pd.DataFrame({
            "s1_id": ["S1-1", None],
            "cand_id": ["S2-1", "S2-2"],
            "score": [0.8, 0.5],
        })
        is_valid, errors = validate_path_b_dataframe(df)
        assert not is_valid
        assert any("null" in err.lower() for err in errors)

    def test_out_of_range_score_fails(self):
        df = pd.DataFrame({
            "s1_id": ["S1-1"],
            "cand_id": ["S2-1"],
            "score": [1.5],  # invalid probability
        })
        is_valid, errors = validate_path_b_dataframe(df)
        assert not is_valid
        assert any("range" in err.lower() for err in errors)


# ---------------------------------------------------------------------------
# Parquet Export and Load Roundtrip Tests
# ---------------------------------------------------------------------------

class TestParquetExportAndLoad:
    def test_export_roundtrip(self, mock_candidate_data):
        cross_scores, bi_cands, gt = mock_candidate_data
        df = build_path_b_feature_dataframe(cross_scores, bi_cands, ground_truth=gt)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            exported_path = export_path_b_parquet(df, output_path=tmp_path, verbose=False)
            assert exported_path.exists()

            loaded_df = load_path_b_features(exported_path, validate=True)
            assert len(loaded_df) == len(df)
            assert list(loaded_df.columns) == list(df.columns)
            assert np.allclose(loaded_df["score"], df["score"])
            assert np.allclose(loaded_df["bi_encoder_score"], df["bi_encoder_score"])
            assert (loaded_df["is_match"] == df["is_match"]).all()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_nonexistent_file_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            load_path_b_features("scores/nonexistent_file_xyz.parquet")

    def test_cross_metrics_export_scores_parquet_compatibility(self, mock_candidate_data):
        cross_scores, bi_cands, _ = mock_candidate_data
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            out = export_scores_parquet(
                scored_candidates=cross_scores,
                output_path=tmp_path,
                bi_encoder_candidates=bi_cands,
            )
            assert out.exists()
            df = pd.read_parquet(out)
            assert "score" in df.columns
            assert "bi_encoder_score" in df.columns
            assert "retrieval_rank" in df.columns
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


# ---------------------------------------------------------------------------
# Downstream Path A Ensemble Compatibility
# ---------------------------------------------------------------------------

class TestDownstreamEnsembleMerge:
    def test_merge_with_mock_path_a_features(self, mock_candidate_data):
        cross_scores, bi_cands, gt = mock_candidate_data
        path_b_df = build_path_b_feature_dataframe(cross_scores, bi_cands, ground_truth=gt)

        # Simulate Path A feature table (string distance features)
        path_a_rows = []
        for s1_id in cross_scores:
            for cand_id, _ in cross_scores[s1_id]:
                path_a_rows.append({
                    "s1_id": s1_id,
                    "cand_id": cand_id,
                    "jaro_winkler": 0.88,
                    "levenshtein_ratio": 0.75,
                    "token_sort_ratio": 0.80,
                })
        path_a_df = pd.DataFrame(path_a_rows)

        # Merge on shared pair contract keys
        merged = path_a_df.merge(path_b_df, on=["s1_id", "cand_id"], how="inner")

        assert len(merged) == len(path_a_df), "Merge lost candidate pairs"
        assert "score" in merged.columns
        assert "bi_encoder_score" in merged.columns
        assert "jaro_winkler" in merged.columns
        assert "is_match" in merged.columns
        # No NaNs introduced
        assert merged.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Demo Mode Test
# ---------------------------------------------------------------------------

class TestB5PipelineDemo:
    def test_demo_executes_cleanly(self):
        demo_df = run_demo()
        assert isinstance(demo_df, pd.DataFrame)
        assert len(demo_df) > 0
        assert "score" in demo_df.columns
        assert "bi_encoder_score" in demo_df.columns
        assert "retrieval_rank" in demo_df.columns
