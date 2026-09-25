"""
tests/test_path_b_b2.py

Unit tests for Path B Ticket B2:
- Hard negative triplet mining
- InputExample validation
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_b.hard_negatives import mine_triplets_from_retrieval


def test_mine_triplets_basic():
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1"},
        {"entity_id": "S1-2"},
        {"entity_id": "S1-3"},  # Singleton
    ])
    corpus_df = pd.DataFrame([
        {"entity_id": "S2-1"},
        {"entity_id": "S2-2"},
        {"entity_id": "S2-3"},
        {"entity_id": "S2-4"},
    ])

    s1_texts = {
        "S1-1": "Acme Corp | 123 Main St",
        "S1-2": "Beta LLC | 456 Elm St",
        "S1-3": "Gamma Inc | 789 Oak St",
    }
    corpus_texts = {
        "S2-1": "Acme Corporation | 123 Main St",   # True match for S1-1
        "S2-2": "Acme Tools | 123 Main St",          # Confuser (hard neg for S1-1)
        "S2-3": "Beta Limited | 456 Elm St",         # True match for S1-2
        "S2-4": "Beta Consulting | 999 Pine St",     # Confuser (hard neg for S1-2)
    }

    ground_truth = {
        "S1-1": ["S2-1"],
        "S1-2": ["S2-3"],
        "S1-3": [],  # Singleton
    }

    retrieval_candidates = {
        "S1-1": [("S2-1", 0.95), ("S2-2", 0.85)],  # S2-2 is hard negative
        "S1-2": [("S2-4", 0.90), ("S2-3", 0.88)],  # S2-4 is hard negative
        "S1-3": [("S2-2", 0.50)],
    }

    examples = mine_triplets_from_retrieval(
        s1_df=s1_df,
        corpus_df=corpus_df,
        s1_texts=s1_texts,
        corpus_texts=corpus_texts,
        ground_truth=ground_truth,
        retrieval_candidates=retrieval_candidates,
        max_negatives_per_positive=1,
    )

    assert len(examples) == 2

    # Check S1-1 triplet
    ex1 = examples[0]
    assert len(ex1.texts) == 3
    assert ex1.texts[0] == "Acme Corp | 123 Main St"
    assert ex1.texts[1] == "Acme Corporation | 123 Main St"
    assert ex1.texts[2] == "Acme Tools | 123 Main St"

    # Check S1-2 triplet
    ex2 = examples[1]
    assert len(ex2.texts) == 3
    assert ex2.texts[0] == "Beta LLC | 456 Elm St"
    assert ex2.texts[1] == "Beta Limited | 456 Elm St"
    assert ex2.texts[2] == "Beta Consulting | 999 Pine St"


def test_mine_triplets_fallback_pairs():
    # If no negative is retrieved (all candidates are true matches) and no other corpus entity exists, falls back to pair
    s1_df = pd.DataFrame([{"entity_id": "S1-1"}])
    corpus_df = pd.DataFrame([{"entity_id": "S2-1"}])

    s1_texts = {"S1-1": "Alpha | NYC"}
    corpus_texts = {"S2-1": "Alpha Inc | NYC"}
    ground_truth = {"S1-1": ["S2-1"]}
    retrieval_candidates = {"S1-1": ["S2-1"]}  # Only true match retrieved

    examples = mine_triplets_from_retrieval(
        s1_df=s1_df,
        corpus_df=corpus_df,
        s1_texts=s1_texts,
        corpus_texts=corpus_texts,
        ground_truth=ground_truth,
        retrieval_candidates=retrieval_candidates,
        fallback_to_random_neg=False,
    )

    assert len(examples) == 1
    assert len(examples[0].texts) == 2  # (anchor, positive) pair


def test_mine_triplets_fallback_random_negative():
    # When no hard negative was retrieved, but other corpus entities exist,
    # fallback_to_random_neg=True samples a non-matching corpus entity as negative
    s1_df = pd.DataFrame([{"entity_id": "S1-1"}])
    corpus_df = pd.DataFrame([{"entity_id": "S2-1"}, {"entity_id": "S2-2"}])

    s1_texts = {"S1-1": "Alpha | NYC"}
    corpus_texts = {"S2-1": "Alpha Inc | NYC", "S2-2": "Omega LLC | London"}
    ground_truth = {"S1-1": ["S2-1"]}
    retrieval_candidates = {"S1-1": ["S2-1"]}  # Only true match retrieved

    examples = mine_triplets_from_retrieval(
        s1_df=s1_df,
        corpus_df=corpus_df,
        s1_texts=s1_texts,
        corpus_texts=corpus_texts,
        ground_truth=ground_truth,
        retrieval_candidates=retrieval_candidates,
        fallback_to_random_neg=True,
    )

    assert len(examples) == 1
    assert len(examples[0].texts) == 3  # (anchor, positive, random_negative)
    assert examples[0].texts[0] == "Alpha | NYC"
    assert examples[0].texts[1] == "Alpha Inc | NYC"
    assert examples[0].texts[2] == "Omega LLC | London"


def test_mine_triplets_max_triplets_cap():
    # Ensures max_triplets_per_anchor caps the number of generated triplets
    s1_df = pd.DataFrame([{"entity_id": "S1-1"}])
    corpus_df = pd.DataFrame([
        {"entity_id": f"S2-{i}"} for i in range(1, 10)
    ])
    s1_texts = {"S1-1": "Alpha | NYC"}
    corpus_texts = {f"S2-{i}": f"Entity {i} | NYC" for i in range(1, 10)}
    ground_truth = {"S1-1": ["S2-1", "S2-2", "S2-3"]}
    retrieval_candidates = {"S1-1": [f"S2-{i}" for i in range(1, 10)]}

    examples = mine_triplets_from_retrieval(
        s1_df=s1_df,
        corpus_df=corpus_df,
        s1_texts=s1_texts,
        corpus_texts=corpus_texts,
        ground_truth=ground_truth,
        retrieval_candidates=retrieval_candidates,
        max_negatives_per_positive=3,
        max_triplets_per_anchor=2,
    )

    assert len(examples) == 2


if __name__ == "__main__":
    test_mine_triplets_basic()
    test_mine_triplets_fallback_pairs()
    test_mine_triplets_fallback_random_negative()
    test_mine_triplets_max_triplets_cap()
    print("All Ticket B2 unit tests passed successfully!")

