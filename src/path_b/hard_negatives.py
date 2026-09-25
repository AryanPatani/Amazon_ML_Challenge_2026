"""
src/path_b/hard_negatives.py

Ticket B2: Hard negative mining for contrastive bi-encoder fine-tuning.
Extracts (anchor, positive, hard_negative) triplets using ground truth matches
and B1 vector retrieval confusers.
"""

from __future__ import annotations

from typing import Optional, Union
import pandas as pd


class InputExample:
    """Lightweight drop-in for sentence_transformers.InputExample."""

    def __init__(self, texts: list[str], guid: str = "", label: Union[int, float] = 0.0) -> None:
        self.texts = texts
        self.guid = guid
        self.label = label

    def __repr__(self) -> str:
        return f"<InputExample> label: {self.label}, texts: {self.texts}"


import numpy as np


def mine_triplets_from_retrieval(
    s1_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    s1_texts: dict[str, str],
    corpus_texts: dict[str, str],
    ground_truth: dict[str, list[str]],
    retrieval_candidates: dict[str, list[Union[str, tuple[str, float]]]],
    max_negatives_per_positive: int = 2,
    min_negative_rank: int = 1,
    max_positives_per_anchor: Optional[int] = 5,
    max_triplets_per_anchor: Optional[int] = 8,
    fallback_to_random_neg: bool = True,
    random_seed: int = 42,
) -> list[InputExample]:
    """Mine contrastive training triplets (anchor, positive, negative).

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source 1 DataFrame with 'entity_id'.
    corpus_df : pd.DataFrame
        Corpus DataFrame (S2 + S3) with 'entity_id'.
    s1_texts : dict[str, str]
        Mapping s1_id -> serialized text string.
    corpus_texts : dict[str, str]
        Mapping corpus_id -> serialized text string.
    ground_truth : dict[str, list[str]]
        Mapping s1_id -> list of true matching corpus entity IDs.
    retrieval_candidates : dict[str, list[str | tuple[str, float]]]
        Mapping s1_id -> list of candidate IDs (or (id, score) tuples) from B1 retrieval.
    max_negatives_per_positive : int, optional
        Maximum number of hard negatives to pair with each positive match, by default 2.
    min_negative_rank : int, optional
        Minimum rank of negative candidates to consider (e.g. 1 means top confuser,
        2 skips top-1 to avoid label noise false positives), by default 1.
    max_positives_per_anchor : Optional[int], optional
        Capping on positive matches per S1 anchor to prevent cluster skew, by default 5.
    max_triplets_per_anchor : Optional[int], optional
        Maximum number of total triplets generated per S1 anchor to ensure balanced training, by default 8.
    fallback_to_random_neg : bool, optional
        If True and no hard negative was retrieved, samples a non-matching corpus entity
        as negative to guarantee uniform 3-tuples (preventing zip truncation in SentenceTransformer),
        by default True.
    random_seed : int, optional
        Random seed for deterministic fallback negative selection, by default 42.

    Returns
    -------
    list[InputExample]
        List of InputExample objects with texts=[anchor, positive, negative]
        (or texts=[anchor, positive] if fallback_to_random_neg=False and no negative was retrieved).
    """
    examples: list[InputExample] = []
    num_triplets = 0
    num_pairs = 0

    # Ensure retrieval candidates are lists of strings
    clean_candidates: dict[str, list[str]] = {}
    for q_id, cands in retrieval_candidates.items():
        if cands and isinstance(cands[0], tuple):
            clean_candidates[q_id] = [c[0] for c in cands]
        else:
            clean_candidates[q_id] = list(cands)

    s1_ids = s1_df["entity_id"].tolist()
    start_neg_idx = max(0, min_negative_rank - 1)

    # Pre-extract corpus IDs for deterministic fallback lookup
    all_corpus_ids = corpus_df["entity_id"].tolist() if "entity_id" in corpus_df.columns else list(corpus_texts.keys())
    rng = np.random.default_rng(random_seed)

    for s1_id in s1_ids:
        anchor_text = s1_texts.get(s1_id)
        if not anchor_text:
            continue

        true_matches = ground_truth.get(s1_id, [])
        if not true_matches:
            # Singletons have no positive pairs in contrastive matching
            continue

        true_set = set(true_matches)
        cands = clean_candidates.get(s1_id, [])

        # Hard negatives: candidates that are NOT in true matches
        hard_negatives = [c for c in cands if c not in true_set]

        # Select negative candidate IDs
        if hard_negatives:
            selected_negs = hard_negatives[start_neg_idx : start_neg_idx + max_negatives_per_positive]
            if not selected_negs:
                selected_negs = hard_negatives[:max_negatives_per_positive]
        elif fallback_to_random_neg and len(all_corpus_ids) > len(true_set):
            # Sample random non-matching negative(s) from corpus to maintain uniform 3-tuples
            fallback_cands: list[str] = []
            for _ in range(max_negatives_per_positive * 5):
                rand_idx = int(rng.integers(0, len(all_corpus_ids)))
                cand_id = all_corpus_ids[rand_idx]
                if cand_id not in true_set and cand_id not in fallback_cands:
                    fallback_cands.append(cand_id)
                    if len(fallback_cands) >= max_negatives_per_positive:
                        break
            selected_negs = fallback_cands
        else:
            selected_negs = []

        # Limit positives per anchor if requested
        active_positives = true_matches[:max_positives_per_anchor] if max_positives_per_anchor else true_matches
        anchor_triplet_count = 0

        for match_id in active_positives:
            pos_text = corpus_texts.get(match_id)
            if not pos_text:
                continue

            if selected_negs:
                for neg_id in selected_negs:
                    neg_text = corpus_texts.get(neg_id)
                    if not neg_text:
                        continue
                    # Skip if negative text identically matches positive or anchor text
                    if neg_text == pos_text or neg_text == anchor_text:
                        continue

                    examples.append(InputExample(texts=[anchor_text, pos_text, neg_text]))
                    num_triplets += 1
                    anchor_triplet_count += 1
                    if max_triplets_per_anchor and anchor_triplet_count >= max_triplets_per_anchor:
                        break
                if max_triplets_per_anchor and anchor_triplet_count >= max_triplets_per_anchor:
                    break
            else:
                # Fallback to (anchor, positive) pair if explicitly requested
                examples.append(InputExample(texts=[anchor_text, pos_text]))
                num_pairs += 1

    print(f"[HardNegatives] Generated {len(examples):,} training examples: "
          f"{num_triplets:,} triplets (with hard/fallback negatives) + {num_pairs:,} pairs.")
    return examples
