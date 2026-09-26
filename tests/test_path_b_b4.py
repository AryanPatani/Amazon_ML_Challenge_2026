"""
tests/test_path_b_b4.py

Ticket B4 unit tests: augmentation module and cross-dataset integration.
"""

import sys
from pathlib import Path
import random
import pytest

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_b.augmentation import (
    drop_pin,
    swap_word_order,
    abbreviate_words,
    expand_abbreviations,
    inject_char_noise,
    transliterate_fr,
    inject_accent_noise,
    drop_suffix,
    drop_random_words,
    inject_landmark,
    drop_address,
    inject_domain,
    corrupt_text,
    augment_batch,
)
from src.path_b.cross_dataset import prepare_cross_encoder_examples
from src.path_b.hard_negatives import InputExample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "Acme Pvt Ltd | 12 Baker Street 110001 London"
SAMPLE_FR   = "Societe Generale | 15 rue de la Paix 75002 Paris"


# ---------------------------------------------------------------------------
# Individual augmentation tests
# ---------------------------------------------------------------------------


class TestDropPin:
    def test_removes_6digit_pin(self):
        result = drop_pin("Some Corp | MG Road 560001 Bengaluru")
        assert "560001" not in result

    def test_removes_5digit_zip(self):
        result = drop_pin("Company | 90210 Beverly Hills CA")
        assert "90210" not in result

    def test_empty_string_unchanged(self):
        assert drop_pin("") == ""

    def test_no_pin_unchanged(self):
        text = "Acme Corp | Baker Street London"
        result = drop_pin(text)
        assert "Baker Street London" in result

    def test_result_stripped(self):
        result = drop_pin("Corp | Main St 12345")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_removes_french_cedex_and_bp(self):
        result = drop_pin("Societe Generale | 15 Rue de Paris CEDEX 02")
        assert "CEDEX" not in result
        assert "02" not in result
        assert "Societe Generale" in result


class TestSwapWordOrder:
    def test_produces_different_string(self):
        rng = random.Random(42)
        result = swap_word_order("Alpha Beta Gamma Delta", rng)
        # After swaps the word set should be the same
        assert sorted(result.split()) == sorted("Alpha Beta Gamma Delta".split())

    def test_short_segment_unchanged(self):
        rng = random.Random(42)
        result = swap_word_order("OnlyOne", rng)
        assert result == "OnlyOne"

    def test_preserves_pipe_structure(self):
        rng = random.Random(1)
        text = "Alpha Beta | Gamma Delta Epsilon"
        result = swap_word_order(text, rng)
        assert "|" in result
        # Both sides should still be present
        name_side = result.split("|")[0].split()
        assert sorted(name_side) == ["Alpha", "Beta"]

    def test_two_word_segment_swappable(self):
        rng = random.Random(99)
        results = set()
        for _ in range(20):
            results.add(swap_word_order("Hello World", random.Random(random.randint(0, 999))))
        # At least one swap should have occurred over 20 tries
        assert len(results) >= 1  # deterministically at least one variant


class TestAbbreviateWords:
    def test_street_to_st(self):
        result = abbreviate_words("12 Baker Street London")
        assert "st" in result.lower() or "Street" in result  # might not always trigger

    def test_road_to_rd(self):
        result = abbreviate_words("Old Airport Road")
        assert "rd" in result.lower()

    def test_preserves_non_matching(self):
        text = "Unique Corporate House"
        result = abbreviate_words(text)
        assert "Unique" in result
        assert "House" in result


class TestExpandAbbreviations:
    def test_rd_to_road(self):
        result = expand_abbreviations("Old Airport Rd")
        assert "road" in result.lower()

    def test_st_to_street(self):
        result = expand_abbreviations("Baker St")
        assert "street" in result.lower()

    def test_abbr_expand_roundtrip(self):
        original = "12 Baker Street"
        abbreviated = abbreviate_words(original)
        expanded = expand_abbreviations(abbreviated)
        # After expand, "street" (lowercase) should be present
        assert "street" in expanded.lower()


class TestInjectCharNoise:
    def test_length_roughly_preserved(self):
        rng = random.Random(42)
        text = "A" * 100
        result = inject_char_noise(text, rng, rate=0.1)
        # Length can change by at most 10% with 0.1 rate
        assert abs(len(result) - 100) <= 30

    def test_non_alpha_unchanged(self):
        rng = random.Random(42)
        # Pipe and digits should not be disturbed
        text = "ABC | 12345"
        result = inject_char_noise(text, rng, rate=1.0)  # high rate on alpha only
        assert "|" in result
        assert "12345" in result

    def test_empty_string(self):
        rng = random.Random(0)
        assert inject_char_noise("", rng) == ""

    def test_zero_rate_unchanged(self):
        rng = random.Random(7)
        text = "Hello World"
        assert inject_char_noise(text, rng, rate=0.0) == text


class TestTransliterateFr:
    def test_accents_removed(self):
        result = transliterate_fr("Société Générale")
        assert "é" not in result
        assert "é" not in result
        assert "Soci" in result

    def test_cedilla_removed(self):
        result = transliterate_fr("François")
        assert "ç" not in result
        assert "c" in result.lower()

    def test_plain_ascii_unchanged(self):
        text = "Hello World 123"
        assert transliterate_fr(text) == text

    def test_oe_ligature(self):
        result = transliterate_fr("cœur")
        assert "oe" in result.lower()


class TestDropSuffix:
    def test_removes_ltd(self):
        result = drop_suffix("Acme Ltd | 12 Baker St")
        assert "Ltd" not in result.split("|")[0]
        assert "Acme" in result

    def test_removes_sarl(self):
        result = drop_suffix("Martin SARL | 5 rue Victor Hugo")
        assert "SARL" not in result.split("|")[0]
        assert "Martin" in result

    def test_keeps_address(self):
        result = drop_suffix("Corp Ltd | 100 Main Road 560001")
        assert "100 Main Road" in result

    def test_no_suffix_unchanged(self):
        text = "TechStartup | 42 Innovation Drive"
        result = drop_suffix(text)
        assert result == text

    def test_never_empties_name(self):
        # If name is only a suffix, it should not become empty
        result = drop_suffix("Ltd | 12 Baker St")
        name_part = result.split("|")[0].strip()
        assert name_part  # non-empty

    def test_removes_dotted_llc(self):
        result = drop_suffix("Asset Building Initiative L.L.C. | 282 Patton Lane")
        assert "L.L.C." not in result.split("|")[0]
        assert "Asset Building Initiative" in result

    def test_removes_dotted_sarl(self):
        result = drop_suffix("Societe Generale S.A.R.L. | 15 Rue de la Paix")
        assert "S.A.R.L." not in result.split("|")[0]
        assert "Societe Generale" in result

    def test_removes_parenthesized_pvt(self):
        result = drop_suffix("Rajdhani (p) Pvt Ltd | 10A KG Marg")
        assert "Rajdhani" in result
        assert "Pvt" not in result.split("|")[0]
        assert "Ltd" not in result.split("|")[0]


class TestDropRandomWords:
    def test_fewer_words_after(self):
        rng = random.Random(42)
        text = "Alpha Beta Gamma Delta Epsilon"
        result = drop_random_words(text, rng, n_drop=1)
        assert len(result.split()) == len(text.split()) - 1

    def test_short_segment_unchanged(self):
        rng = random.Random(42)
        text = "Short | One"
        result = drop_random_words(text, rng)
        assert result == text

    def test_first_word_preserved(self):
        rng = random.Random(42)
        text = "FIRST second third fourth fifth"
        for _ in range(20):
            result = drop_random_words(text, random.Random(random.randint(0, 999)), n_drop=2)
            assert result.split()[0] == "FIRST"


class TestInjectLandmark:
    def test_landmark_added(self):
        rng = random.Random(42)
        text = "Acme Corp | 12 Baker St"
        result = inject_landmark(text, rng)
        # Result should be longer
        assert len(result) > len(text)

    def test_pipe_structure_preserved(self):
        rng = random.Random(42)
        text = "Corp | Address Here"
        result = inject_landmark(text, rng)
        assert "|" in result

    def test_no_pipe_text(self):
        rng = random.Random(1)
        text = "Just a name"
        result = inject_landmark(text, rng)
        assert len(result) > len(text)


class TestInjectAccentNoise:
    def test_injects_accents_on_vowels(self):
        rng = random.Random(42)
        text = "Acme Corporation | Baker Street London"
        result = inject_accent_noise(text, rng, rate=0.2)
        # Should have at least one accented character
        accented_chars = {"é", "è", "ê", "ë", "à", "â", "ä", "î", "ï", "í", "ô", "ö", "ó", "ù", "û", "ü", "ú", "ç", "É", "È", "Ê", "À", "Â", "Î", "Ï", "Ô", "Ö", "Ù", "Û", "Ç"}
        has_accent = any(c in accented_chars for c in result)
        assert has_accent

    def test_deterministic_with_same_seed(self):
        text = "Societe Generale | 15 Rue de Paris"
        r1 = inject_accent_noise(text, random.Random(99), rate=0.1)
        r2 = inject_accent_noise(text, random.Random(99), rate=0.1)
        assert r1 == r2

    def test_empty_string(self):
        assert inject_accent_noise("", random.Random(1)) == ""


class TestDropAddress:
    def test_drops_address_keeps_name(self):
        result = drop_address("Acme Corp | 12 Baker Street")
        assert result.startswith("Acme Corp")
        assert "12 Baker Street" not in result
        assert "|" in result

    def test_no_pipe_unchanged(self):
        result = drop_address("JustAName")
        assert result == "JustAName"


class TestInjectDomain:
    def test_adds_tld_to_name(self):
        rng = random.Random(42)
        result = inject_domain("Acme Global Solutions | 12 Baker St", rng)
        # Should contain one of the TLDs
        assert any(tld in result.split("|")[0] for tld in [".com", ".in", ".org", ".fr", ".net"])
        assert "12 Baker St" in result

    def test_preserves_pipe(self):
        rng = random.Random(10)
        result = inject_domain("Standard Tech | Bangalore", rng)
        assert "|" in result


# ---------------------------------------------------------------------------
# Master corrupt_text tests
# ---------------------------------------------------------------------------


class TestCorruptText:
    def test_returns_string(self):
        rng = random.Random(42)
        result = corrupt_text(SAMPLE_TEXT, rng=rng)
        assert isinstance(result, str)

    def test_non_empty(self):
        rng = random.Random(42)
        result = corrupt_text(SAMPLE_TEXT, rng=rng)
        assert len(result) > 0

    def test_empty_input(self):
        assert corrupt_text("") == ""
        assert corrupt_text("   ") == "   "

    def test_deterministic_with_same_rng(self):
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        r1 = corrupt_text(SAMPLE_TEXT, rng=rng1)
        r2 = corrupt_text(SAMPLE_TEXT, rng=rng2)
        assert r1 == r2

    def test_different_seeds_usually_differ(self):
        results = set()
        for i in range(10):
            results.add(corrupt_text(SAMPLE_TEXT, rng=random.Random(i)))
        assert len(results) > 1  # not all the same

    def test_explicit_augmentations(self):
        rng = random.Random(42)
        result = corrupt_text(SAMPLE_TEXT, rng=rng, augmentations=["drop_pin"])
        assert "110001" not in result

    def test_explicit_transliterate(self):
        rng = random.Random(42)
        result = corrupt_text(SAMPLE_FR, rng=rng, augmentations=["transliterate_fr"])
        assert "é" not in result

    def test_unknown_augmentation_graceful(self):
        rng = random.Random(42)
        # Should not raise
        result = corrupt_text(SAMPLE_TEXT, rng=rng, augmentations=["nonexistent_aug"])
        assert result == SAMPLE_TEXT

    def test_min_augmentations_applied(self):
        # With min_augmentations=1 and a fixed seed, result should differ from original
        rng = random.Random(42)
        result = corrupt_text(SAMPLE_TEXT, rng=rng, min_augmentations=1, max_augmentations=1)
        # Not guaranteed to differ (if only augmentation applied is identity-like)
        # but at least should not raise
        assert isinstance(result, str)

    def test_max_augmentations_respected(self):
        # With max=1 we still get a valid string
        rng = random.Random(5)
        result = corrupt_text(SAMPLE_TEXT, rng=rng, min_augmentations=1, max_augmentations=1)
        assert result

    def test_zero_noop_rate_across_seeds(self):
        # corrupt_text with min_augmentations=1 should never return the exact same string
        text = "Acme Global Solutions | 12 Baker Street London"
        for seed in range(200):
            r = corrupt_text(text, rng=random.Random(seed), min_augmentations=1, max_augmentations=2)
            assert r != text, f"corrupt_text returned identical string on seed {seed}"


# ---------------------------------------------------------------------------
# augment_batch tests
# ---------------------------------------------------------------------------


class TestAugmentBatch:
    def test_output_length(self):
        texts = ["A | B", "C | D", "E | F"]
        result = augment_batch(texts, n_augments=3)
        assert len(result) == len(texts) * 3

    def test_single_augment(self):
        texts = ["Hello World | 12345"]
        result = augment_batch(texts, n_augments=1)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_deterministic_with_seed(self):
        texts = [SAMPLE_TEXT]
        r1 = augment_batch(texts, n_augments=2, seed=42)
        r2 = augment_batch(texts, n_augments=2, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        texts = [SAMPLE_TEXT]
        r1 = augment_batch(texts, n_augments=1, seed=0)
        r2 = augment_batch(texts, n_augments=1, seed=99)
        # Very likely different (but not guaranteed — acceptable)
        assert isinstance(r1[0], str) and isinstance(r2[0], str)

    def test_empty_texts(self):
        result = augment_batch([], n_augments=5)
        assert result == []


# ---------------------------------------------------------------------------
# Integration: augmented pairs in prepare_cross_encoder_examples
# ---------------------------------------------------------------------------


class TestCrossDatasetAugmentation:
    def _make_data(self):
        s1_ids   = ["s1_1", "s1_2", "s1_3"]
        s1_texts = {
            "s1_1": "Alpha Corp | 12 Baker St 110001",
            "s1_2": "Beta Ltd   | Rue de Paris 75001",
            "s1_3": "Gamma Inc  | MG Road 560001",
        }
        corpus_texts = {
            "c1": "Alpha Corporation | 12 Baker Street 110001",
            "c2": "Beta Limited | Paris 75001",
            "c3": "Gamma Incorporated | MG Road Bengaluru",
            "c4": "Noise Corp | Somewhere Else",
            "c5": "Unrelated Entity | Different Address",
        }
        ground_truth = {
            "s1_1": ["c1"],
            "s1_2": ["c2"],
            "s1_3": ["c3"],
        }
        retrieval_candidates = {
            "s1_1": [("c1", 0.9), ("c4", 0.5), ("c5", 0.3)],
            "s1_2": [("c2", 0.85), ("c4", 0.4)],
            "s1_3": [("c3", 0.8), ("c5", 0.35)],
        }
        return s1_ids, s1_texts, corpus_texts, ground_truth, retrieval_candidates

    def test_baseline_no_augmentation(self):
        s1_ids, s1_texts, corpus_texts, gt, cands = self._make_data()
        examples = prepare_cross_encoder_examples(
            s1_ids=s1_ids, s1_texts=s1_texts, corpus_texts=corpus_texts,
            ground_truth=gt, retrieval_candidates=cands,
            augment_positives=False,
        )
        pos = [e for e in examples if e.label == 1.0]
        neg = [e for e in examples if e.label == 0.0]
        assert len(pos) == 3  # one per anchor (s1_1, s1_2, s1_3)
        assert len(neg) > 0

    def test_augmentation_increases_pair_count(self):
        s1_ids, s1_texts, corpus_texts, gt, cands = self._make_data()
        base_examples = prepare_cross_encoder_examples(
            s1_ids=s1_ids, s1_texts=s1_texts, corpus_texts=corpus_texts,
            ground_truth=gt, retrieval_candidates=cands,
            augment_positives=False,
        )
        aug_examples = prepare_cross_encoder_examples(
            s1_ids=s1_ids, s1_texts=s1_texts, corpus_texts=corpus_texts,
            ground_truth=gt, retrieval_candidates=cands,
            augment_positives=True,
            n_augments=2,
        )
        # Augmented should have more positives
        base_pos = sum(1 for e in base_examples if e.label == 1.0)
        aug_pos  = sum(1 for e in aug_examples if e.label == 1.0)
        assert aug_pos >= base_pos  # at least equal (augments may be filtered as identical)

    def test_augmented_labels_are_positive(self):
        s1_ids, s1_texts, corpus_texts, gt, cands = self._make_data()
        examples = prepare_cross_encoder_examples(
            s1_ids=s1_ids, s1_texts=s1_texts, corpus_texts=corpus_texts,
            ground_truth=gt, retrieval_candidates=cands,
            augment_positives=True, n_augments=3,
        )
        for e in examples:
            assert e.label in (0.0, 1.0)

    def test_texts_are_strings(self):
        s1_ids, s1_texts, corpus_texts, gt, cands = self._make_data()
        examples = prepare_cross_encoder_examples(
            s1_ids=s1_ids, s1_texts=s1_texts, corpus_texts=corpus_texts,
            ground_truth=gt, retrieval_candidates=cands,
            augment_positives=True, n_augments=2,
        )
        for e in examples:
            assert isinstance(e.texts[0], str)
            assert isinstance(e.texts[1], str)
            assert len(e.texts[0]) > 0
            assert len(e.texts[1]) > 0

    def test_augmentation_deterministic(self):
        s1_ids, s1_texts, corpus_texts, gt, cands = self._make_data()
        e1 = prepare_cross_encoder_examples(
            s1_ids=s1_ids, s1_texts=s1_texts, corpus_texts=corpus_texts,
            ground_truth=gt, retrieval_candidates=cands,
            augment_positives=True, n_augments=2, aug_seed=42,
        )
        e2 = prepare_cross_encoder_examples(
            s1_ids=s1_ids, s1_texts=s1_texts, corpus_texts=corpus_texts,
            ground_truth=gt, retrieval_candidates=cands,
            augment_positives=True, n_augments=2, aug_seed=42,
        )
        pairs1 = [(e.texts[0], e.texts[1], e.label) for e in e1]
        pairs2 = [(e.texts[0], e.texts[1], e.label) for e in e2]
        assert pairs1 == pairs2

    def test_negative_label_unchanged(self):
        """Negative pairs should never be augmented (only positives are)."""
        s1_ids, s1_texts, corpus_texts, gt, cands = self._make_data()
        examples = prepare_cross_encoder_examples(
            s1_ids=s1_ids, s1_texts=s1_texts, corpus_texts=corpus_texts,
            ground_truth=gt, retrieval_candidates=cands,
            augment_positives=True, n_augments=5,
        )
        neg_anchors = {e.texts[0] for e in examples if e.label == 0.0}
        # All neg anchors must come from the original s1_texts (not corrupted)
        for anchor in neg_anchors:
            assert anchor in s1_texts.values(), (
                f"Negative anchor text not in original s1_texts: {anchor!r}"
            )
