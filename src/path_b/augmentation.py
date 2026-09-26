"""
src/path_b/augmentation.py

Ticket B4: Synthetic noise augmentation for cross-encoder robustness.
Generates realistic corruption patterns (drop PIN, swap word order, abbreviate,
transliterate, inject accents, drop suffixes, inject landmarks, missing addresses,
domain-style naming) to improve model generalization across unseen countries (e.g. France).
"""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Callable, Optional, Sequence


_ABBR_TO_FULL: dict = {
    # US / UK
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place",
    "sq": "square", "hwy": "highway", "pkwy": "parkway", "fwy": "freeway",
    "expy": "expressway", "n": "north", "s": "south", "e": "east", "w": "west",
    "nw": "northwest", "ne": "northeast", "sw": "southwest", "se": "southeast",
    "apt": "apartment", "ste": "suite", "bldg": "building", "dept": "department",
    "no": "number",
    # France
    "bd": "boulevard", "av": "avenue", "all": "allee", "r": "rue",
    "imp": "impasse", "rt": "route", "rte": "route", "res": "residence",
    "bat": "batiment", "ch": "chemin", "za": "zone artisanale",
    "zi": "zone industrielle", "zac": "zone d amenagement concerte",
    # India
    "soc": "society", "nagar": "ngr", "marg": "road", "opp": "opposite",
    "nr": "near", "flr": "floor", "stn": "station",
}
_FULL_TO_ABBR: dict = {v: k for k, v in _ABBR_TO_FULL.items()}
# Ensure canonical primary abbreviations are preserved
_FULL_TO_ABBR["road"] = "rd"
_FULL_TO_ABBR["street"] = "st"
_FULL_TO_ABBR["avenue"] = "ave"
_FULL_TO_ABBR["boulevard"] = "blvd"

_LEGAL_SUFFIXES: frozenset = frozenset({
    "ltd", "limited", "pvt", "private", "inc", "incorporated",
    "corp", "corporation", "llc", "llp", "lp", "plc",
    "sa", "sas", "sarl", "srl", "bv", "nv", "gmbh", "ag",
    "oy", "ab", "as", "aps", "co", "company", "group",
    "holdings", "enterprises", "industries", "international",
    "solutions", "services", "technologies", "tech",
    # French & regional corporate acronyms
    "eurl", "sasu", "sci", "snc", "gie", "selarl", "p",
})

_LANDMARKS: tuple = (
    "zone industrielle", "zone artisanale", "parc d activites",
    "centre commercial", "bat a", "bat b", "hall 1", "hall 2",
    "immeuble le dome", "immeuble le galilee", "lotissement les oliviers",
    "quartier les halles", "lieu dit les pins",
)

_ACCENT_MAP: dict = {
    "\u00e0": "a", "\u00e2": "a", "\u00e4": "a", "\u00e1": "a", "\u00e3": "a",
    "\u00e8": "e", "\u00ea": "e", "\u00eb": "e", "\u00e9": "e",
    "\u00ec": "i", "\u00ee": "i", "\u00ef": "i", "\u00ed": "i",
    "\u00f2": "o", "\u00f4": "o", "\u00f6": "o", "\u00f3": "o", "\u00f5": "o",
    "\u00f9": "u", "\u00fb": "u", "\u00fc": "u", "\u00fa": "u",
    "\u00f1": "n", "\u00e7": "c", "\u00fd": "y", "\u00ff": "y",
    "\u0153": "oe", "\u00e6": "ae", "\u00df": "ss",
    "\u00c0": "A", "\u00c2": "A", "\u00c4": "A", "\u00c1": "A",
    "\u00c8": "E", "\u00ca": "E", "\u00cb": "E", "\u00c9": "E",
    "\u00ce": "I", "\u00cf": "I", "\u00cd": "I",
    "\u00d4": "O", "\u00d6": "O", "\u00d3": "O",
    "\u00db": "U", "\u00dc": "U", "\u00da": "U",
    "\u00d1": "N", "\u00c7": "C",
}

_ACCENT_INJECTIONS: dict[str, tuple[str, ...]] = {
    "e": ("\u00e9", "\u00e8", "\u00ea", "\u00eb"),
    "a": ("\u00e0", "\u00e2", "\u00e4"),
    "i": ("\u00ee", "\u00ef", "\u00ed"),
    "o": ("\u00f4", "\u00f6", "\u00f3"),
    "u": ("\u00f9", "\u00fb", "\u00fc", "\u00fa"),
    "c": ("\u00e7",),
    "E": ("\u00c9", "\u00c8", "\u00ca"),
    "A": ("\u00c0", "\u00c2"),
    "I": ("\u00ce", "\u00cf"),
    "O": ("\u00d4", "\u00d6"),
    "U": ("\u00db", "\u00dc"),
    "C": ("\u00c7",),
}

_TLDS: tuple = (".com", ".in", ".org", ".fr", ".co.in", ".net")

_PIN_PATTERN = re.compile(
    r"\b(?:\d{6}|\d{5}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|\d{4,7}|cedex(?:\s*\d+)?|bp\s*\d+|cs\s*\d+)\b",
    re.IGNORECASE,
)


def drop_pin(text: str) -> str:
    """Strip 4-7 digit postal/PIN codes, UK postcodes, and French CEDEX/BP notations."""
    cleaned = _PIN_PATTERN.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if cleaned else text


def swap_word_order(text: str, rng: random.Random, max_swaps: int = 2) -> str:
    """Randomly swap adjacent words within pipe-delimited fields."""
    segments = text.split("|")
    result_segs = []
    for seg in segments:
        words = seg.split()
        if len(words) < 2:
            result_segs.append(seg.strip())
            continue
        swapped = words[:]
        for _ in range(min(max_swaps, len(swapped) - 1)):
            i = rng.randint(0, len(swapped) - 2)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        result_segs.append(" ".join(swapped))
    return " | ".join(result_segs)


def abbreviate_words(text: str) -> str:
    """Replace full street/place words with standard abbreviations."""
    if not _FULL_TO_ABBR:
        return text
    def _replace(m: re.Match) -> str:
        abbr = _FULL_TO_ABBR.get(m.group(0).lower())
        return abbr if abbr else m.group(0)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(_FULL_TO_ABBR, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(_replace, text)


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations to their full word form."""
    if not _ABBR_TO_FULL:
        return text
    def _replace(m: re.Match) -> str:
        full = _ABBR_TO_FULL.get(m.group(0).lower())
        return full if full else m.group(0)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBR_TO_FULL, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(_replace, text)


def inject_char_noise(text: str, rng: random.Random, rate: float = 0.04) -> str:
    """Simulate OCR / typing errors via random character swap, deletion, or insertion."""
    chars = list(text)
    n = len(chars)
    i = 0
    while i < n:
        if chars[i].isalpha() and rng.random() < rate:
            op = rng.randint(0, 2)
            if op == 0 and i < n - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 2
                continue
            elif op == 1:
                chars.pop(i)
                n -= 1
                continue
            else:
                chars.insert(i + 1, chars[i])
                n += 1
                i += 2
                continue
        i += 1
    return "".join(chars)


def transliterate_fr(text: str) -> str:
    """Map accented French/European characters to plain ASCII equivalents."""
    result = []
    for ch in text:
        mapped = _ACCENT_MAP.get(ch)
        if mapped is not None:
            result.append(mapped)
        else:
            decomposed = unicodedata.normalize("NFD", ch)
            ascii_chars = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
            result.append(ascii_chars if ascii_chars else ch)
    return "".join(result)


def inject_accent_noise(text: str, rng: random.Random, rate: float = 0.08) -> str:
    """Randomly replace ASCII vowels and 'c' with accented variants (French / typo style)."""
    chars = list(text)
    modified = False
    for i, ch in enumerate(chars):
        if ch in _ACCENT_INJECTIONS and rng.random() < rate:
            chars[i] = rng.choice(_ACCENT_INJECTIONS[ch])
            modified = True
    if not modified and rate > 0:
        candidates = [i for i, ch in enumerate(chars) if ch in _ACCENT_INJECTIONS]
        if candidates:
            idx = rng.choice(candidates)
            chars[idx] = rng.choice(_ACCENT_INJECTIONS[chars[idx]])
    return "".join(chars)


def _normalize_suffix_word(word: str) -> str:
    """Normalize acronym or suffix word by removing dots, parentheses, and hyphens."""
    return re.sub(r"[\.()\-]", "", word).strip().lower()


def drop_suffix(text: str) -> str:
    """Strip legal corporate form suffixes (Ltd, SARL, L.L.C., GmbH, etc.) from the name."""
    parts = text.split("|", maxsplit=1)
    name_part = parts[0].strip()
    words = name_part.split()
    while words and _normalize_suffix_word(words[-1]) in _LEGAL_SUFFIXES:
        words.pop()
    clean_name = " ".join(words).strip() or name_part
    rest = [p.strip() for p in parts[1:]]
    return " | ".join([clean_name] + rest)


def drop_random_words(text: str, rng: random.Random, n_drop: int = 1) -> str:
    """Drop 1-2 interior words per field while preserving the anchor word."""
    segments = text.split("|")
    result_segs = []
    for seg in segments:
        words = seg.split()
        if len(words) <= 2:
            result_segs.append(seg.strip())
            continue
        interior = words[1:]
        drop_indices = set(rng.sample(range(len(interior)), min(n_drop, len(interior) - 1)))
        filtered = words[:1] + [w for idx, w in enumerate(interior) if idx not in drop_indices]
        result_segs.append(" ".join(filtered))
    return " | ".join(result_segs)


def inject_landmark(text: str, rng: random.Random) -> str:
    """Append a French industrial zone, commercial centre, or building name to the address."""
    landmark = rng.choice(_LANDMARKS)
    parts = [p.strip() for p in text.split("|")]
    if len(parts) >= 2:
        parts[1] = (parts[1] + " " + landmark).strip()
    else:
        parts[0] = (parts[0] + " " + landmark).strip()
    return " | ".join(parts)


def drop_address(text: str) -> str:
    """Simulate missing or empty address fields (observed in ~3.4% of S2/S3 records in EDA)."""
    parts = text.split("|", maxsplit=1)
    if len(parts) >= 2:
        return parts[0].strip() + " |"
    return text


def inject_domain(text: str, rng: random.Random) -> str:
    """Simulate domain/URL business name noise (observed in S3 records in EDA)."""
    parts = text.split("|", maxsplit=1)
    name = parts[0].strip()
    words = name.split()
    if not words:
        return text
    cleaned_name = re.sub(r"[^a-zA-Z0-9]", "", "".join(words)).lower()
    if len(cleaned_name) < 3:
        return text
    tld = rng.choice(_TLDS)
    domain_name = f"www.{cleaned_name}{tld}" if rng.random() < 0.3 else f"{cleaned_name}{tld}"
    if len(parts) >= 2:
        return f"{domain_name} | {parts[1].strip()}"
    return domain_name


_AUG_REGISTRY: list = [
    (drop_pin, 0.4),
    (swap_word_order, 0.3),
    (abbreviate_words, 0.25),
    (expand_abbreviations, 0.2),
    (inject_char_noise, 0.35),
    (transliterate_fr, 0.3),
    (inject_accent_noise, 0.35),
    (drop_suffix, 0.3),
    (drop_random_words, 0.25),
    (inject_landmark, 0.2),
    (drop_address, 0.1),
    (inject_domain, 0.1),
]

_RNG_FUNS = frozenset({
    swap_word_order, inject_char_noise, inject_accent_noise,
    drop_random_words, inject_landmark, inject_domain,
})


_MUTUAL_EXCLUSIONS: dict = {
    inject_accent_noise: {transliterate_fr},
    transliterate_fr: {inject_accent_noise},
    abbreviate_words: {expand_abbreviations},
    expand_abbreviations: {abbreviate_words},
}


def corrupt_text(
    text: str,
    rng: Optional[random.Random] = None,
    augmentations: Optional[Sequence[str]] = None,
    min_augmentations: int = 1,
    max_augmentations: int = 3,
) -> str:
    """Apply a random subset of augmentations to *text*."""
    if rng is None:
        rng = random.Random()
    if not text or not text.strip():
        return text

    _NAME_TO_FN: dict = {
        "drop_pin": drop_pin,
        "swap_word_order": swap_word_order,
        "abbreviate_words": abbreviate_words,
        "expand_abbreviations": expand_abbreviations,
        "inject_char_noise": inject_char_noise,
        "transliterate_fr": transliterate_fr,
        "inject_accent_noise": inject_accent_noise,
        "drop_suffix": drop_suffix,
        "drop_random_words": drop_random_words,
        "inject_landmark": inject_landmark,
        "drop_address": drop_address,
        "inject_domain": inject_domain,
    }

    if augmentations is not None:
        result = text
        for name in augmentations:
            fn = _NAME_TO_FN.get(name)
            if fn is None:
                continue
            try:
                candidate = fn(result, rng) if fn in _RNG_FUNS else fn(result)
                if candidate:
                    result = candidate
            except Exception:
                pass
        return result or text

    result = text
    applied = 0
    applied_fns: set = set()
    registry_order = list(range(len(_AUG_REGISTRY)))
    rng.shuffle(registry_order)

    for idx in registry_order:
        fn, prob = _AUG_REGISTRY[idx]
        if applied >= max_augmentations:
            break
        # Skip if an inverse operation was already applied in this pass
        if any(fn in _MUTUAL_EXCLUSIONS.get(prev, set()) for prev in applied_fns):
            continue
        if applied < min_augmentations or rng.random() < prob:
            try:
                candidate = fn(result, rng) if fn in _RNG_FUNS else fn(result)
                if candidate != result:
                    result = candidate
                    applied += 1
                    applied_fns.add(fn)
            except Exception:
                pass

    # If min_augmentations requested but net result is unchanged, guarantee a change
    if (applied < min_augmentations or result == text) and len(result.strip()) >= 2:
        try:
            cand = inject_char_noise(result, rng, rate=0.15)
            if cand != text:
                result = cand
                applied += 1
            else:
                chars = list(result)
                alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
                if len(alpha_indices) >= 2:
                    k = rng.choice(alpha_indices[:-1])
                    chars[k], chars[k + 1] = chars[k + 1], chars[k]
                    result = "".join(chars)
                    applied += 1
        except Exception:
            pass

    return result or text


def augment_batch(
    texts: list,
    n_augments: int = 1,
    seed: int = 42,
    min_augmentations: int = 1,
    max_augmentations: int = 3,
) -> list:
    """Generate *n_augments* corrupted variants for each text in *texts*."""
    master_rng = random.Random(seed)
    results = []
    for text in texts:
        sub_seed = master_rng.randint(0, 2**31 - 1)
        for j in range(n_augments):
            aug_rng = random.Random(sub_seed + j * 1_000_003)
            results.append(
                corrupt_text(
                    text,
                    rng=aug_rng,
                    min_augmentations=min_augmentations,
                    max_augmentations=max_augmentations,
                )
            )
    return results
