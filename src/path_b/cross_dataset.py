"""
src/path_b/cross_dataset.py

Ticket B3/B4: Pair dataset preparation for Cross-Encoder re-ranking.
Extracts labeled (anchor, candidate) pairs from retrieval candidate pools,
balancing true matches (label=1.0) and retrieved hard negative confusers (label=0.0).

Ticket B4 adds synthetic noise augmentation: each positive pair is replicated
N times with probabilistically corrupted anchor/candidate text (drop PIN,
swap word order, abbreviate, transliterate accents, drop suffix, etc.) to
simulate unseen-country noise and improve model generalisation.
"""

from __future__ import annotations

import random
from typing import Optional, Union
import numpy as np

from src.path_b.hard_negatives import InputExample
from src.path_b.augmentation import corrupt_text


def prepare_cross_encoder_examples(
    s1_ids: list[str],
    s1_texts: dict[str, str],
    corpus_texts: dict[str, str],
    ground_truth: dict[str, list[str]],
    retrieval_candidates: dict[str, list[Union[str, tuple[str, float]]]],
    max_negatives_per_positive: int = 4,
    max_positives_per_anchor: Optional[int] = 5,
    max_negatives_per_singleton: int = 2,
    include_fallback_negatives: bool = True,
    corpus_ids: Optional[list[str]] = None,
    random_seed: int = 42,
    # B4 Augmentation parameters
    augment_positives: bool = False,
    n_augments: int = 2,
    aug_min_ops: int = 1,
    aug_max_ops: int = 3,
    aug_seed: int = 99,
) -> list[InputExample]:
    """Build labeled (S1, candidate) pairs for cross-encoder training.

    Parameters
    ----------
    s1_ids : list[str]
        List of S1 entity IDs to include in the training set.
    s1_texts : dict[str, str]
        Mapping s1_id -> serialized text string.
    corpus_texts : dict[str, str]
        Mapping corpus_id -> serialized text string.
    ground_truth : dict[str, list[str]]
        Mapping s1_id -> list of true matching corpus entity IDs.
    retrieval_candidates : dict[str, list[Union[str, tuple[str, float]]]]
        Mapping s1_id -> list of candidate IDs (or (id, score) tuples) from retrieval.
    max_negatives_per_positive : int, optional
        Number of hard negative candidates to pair with each positive, by default 4.
    max_positives_per_anchor : Optional[int], optional
        Maximum positive matches per anchor to prevent cluster skew, by default 5.
    max_negatives_per_singleton : int, optional
        Number of negative candidates to sample for singletons (anchors with 0 matches), by default 2.
    include_fallback_negatives : bool, optional
        Sample random negatives from corpus if no hard negative was retrieved, by default True.
    corpus_ids : Optional[list[str]], optional
        Pool of corpus IDs for random fallback negatives.
    random_seed : int, optional
        Random seed for deterministic negative sampling, by default 42.
    augment_positives : bool, optional
        (B4) If True, generate *n_augments* noisy variants of every positive pair.
        Corrupted anchor / corrupted candidate pairs expose the cross-encoder to
        unseen-country surface-level noise (PIN drop, word-order swap, French
        accent transliteration, abbreviations, etc.).
        By default False.
    n_augments : int, optional
        Number of noisy copies per positive pair when augment_positives=True.
        By default 2.
    aug_min_ops : int, optional
        Minimum number of augmentation operations applied per copy, by default 1.
    aug_max_ops : int, optional
        Maximum number of augmentation operations applied per copy, by default 3.
    aug_seed : int, optional
        Seed for the augmentation RNG (separate from negative-sampling seed).
        By default 99.

    Returns
    -------
    list[InputExample]
        List of InputExample with texts=[s1_text, cand_text] and label in {0.0, 1.0}.
    """
    examples: list[InputExample] = []
    num_pos = 0
    num_neg = 0
    num_aug = 0

    rng = np.random.default_rng(random_seed)
    fallback_pool = corpus_ids or list(corpus_texts.keys())

    # B4: augmentation RNG (separate from negative sampling RNG for reproducibility)
    aug_rng = random.Random(aug_seed) if augment_positives else None

    # Clean candidates to string IDs
    clean_candidates: dict[str, list[str]] = {}
    for q_id, cands in retrieval_candidates.items():
        if cands and isinstance(cands[0], tuple):
            clean_candidates[q_id] = [c[0] for c in cands]
        else:
            clean_candidates[q_id] = list(cands)

    for s1_id in s1_ids:
        anchor_text = s1_texts.get(s1_id)
        if not anchor_text:
            continue

        true_matches = ground_truth.get(s1_id, [])
        true_set = set(true_matches)
        cands = clean_candidates.get(s1_id, [])

        # 1. Positive Pairs
        active_positives = true_matches[:max_positives_per_anchor] if max_positives_per_anchor else true_matches
        pos_added_for_anchor = 0
        pos_texts_for_anchor: set[str] = set()
        for match_id in active_positives:
            pos_text = corpus_texts.get(match_id)
            if pos_text:
                examples.append(InputExample(texts=[anchor_text, pos_text], label=1.0))
                num_pos += 1
                pos_added_for_anchor += 1
                pos_texts_for_anchor.add(pos_text)

                # B4: augmented positive copies (corrupted anchor + corrupted candidate)
                if augment_positives and aug_rng is not None:
                    for _ in range(n_augments):
                        # Independently corrupt each side to maximise coverage
                        aug_anchor = corrupt_text(
                            anchor_text, rng=aug_rng,
                            min_augmentations=aug_min_ops,
                            max_augmentations=aug_max_ops,
                        )
                        aug_cand = corrupt_text(
                            pos_text, rng=aug_rng,
                            min_augmentations=aug_min_ops,
                            max_augmentations=aug_max_ops,
                        )
                        # Skip if augmentation produced an identical pair
                        if aug_anchor == anchor_text and aug_cand == pos_text:
                            continue
                        examples.append(InputExample(texts=[aug_anchor, aug_cand], label=1.0))
                        num_aug += 1

        # 2. Hard Negative Pairs
        hard_negs = [c for c in cands if c not in true_set]

        # Calculate quota of negatives for this anchor
        if pos_added_for_anchor > 0:
            target_negs = max(1, pos_added_for_anchor * max_negatives_per_positive)
        else:
            target_negs = max_negatives_per_singleton

        selected_negs: list[str] = []
        if hard_negs:
            selected_negs = hard_negs[:target_negs]

        # If hard negatives are fewer than quota, sample random non-matching negatives if enabled
        if len(selected_negs) < target_negs and include_fallback_negatives and len(fallback_pool) > len(true_set):
            needed = target_negs - len(selected_negs)
            for _ in range(needed * 5):
                rand_idx = int(rng.integers(0, len(fallback_pool)))
                rand_id = fallback_pool[rand_idx]
                if rand_id not in true_set and rand_id not in selected_negs:
                    selected_negs.append(rand_id)
                    if len(selected_negs) >= target_negs:
                        break

        for neg_id in selected_negs:
            neg_text = corpus_texts.get(neg_id)
            if not neg_text:
                continue
            # Guard against exact text identity false signals or conflicting positive matches
            if neg_text == anchor_text or neg_text in pos_texts_for_anchor:
                continue

            examples.append(InputExample(texts=[anchor_text, neg_text], label=0.0))
            num_neg += 1

    aug_msg = f" + {num_aug:,} augmented positives (B4)" if augment_positives else ""
    print(f"[CrossDataset] Generated {len(examples):,} cross-encoder training pairs: "
          f"{num_pos:,} positives (1.0) + {num_neg:,} negatives (0.0){aug_msg}.")
    return examples



def build_inference_pairs(
    candidate_dict: dict[str, list[Union[str, tuple[str, float]]]],
    s1_texts: dict[str, str],
    corpus_texts: dict[str, str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Flatten candidate pairs into parallel lists for batched model inference.

    Parameters
    ----------
    candidate_dict : dict[str, list[str | tuple[str, float]]]
        Mapping s1_id -> candidate IDs.
    s1_texts : dict[str, str]
        Mapping s1_id -> serialized text.
    corpus_texts : dict[str, str]
        Mapping corpus_id -> serialized text.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]]]
        - text_pairs: list of (s1_text, cand_text) tuples for model prediction.
        - id_pairs: list of (s1_id, cand_id) tuples corresponding to text_pairs.
    """
    text_pairs: list[tuple[str, str]] = []
    id_pairs: list[tuple[str, str]] = []

    for s1_id, cands in candidate_dict.items():
        s1_text = s1_texts.get(s1_id)
        if not s1_text:
            continue

        for item in cands:
            cand_id = item[0] if isinstance(item, tuple) else item
            cand_text = corpus_texts.get(cand_id)
            if not cand_text:
                continue

            text_pairs.append((s1_text, cand_text))
            id_pairs.append((s1_id, cand_id))

    return text_pairs, id_pairs
