"""
src/common/normalize.py

Shared text normalizer for Business Entity Resolution (Ticket 0.3).

Provides:
    normalize_name(raw: str) -> dict
    normalize_address(raw: str) -> dict

Both are pure functions: same input always gives same output, no country
branching, no external lookups. Designed to be vectorization-friendly via
pandas .apply().

Usage:
    from normalize import normalize_name, normalize_address

    out = normalize_name("Standard Automation Consultants Inc")
    out = normalize_address("1671 442, Sweetwater, TX")
"""

import re
import unicodedata

NORMALIZER_VERSION = "0.1"

# --------------------------------------------------------------------------- #
# Lookup tables — extend these as you find more patterns in the training data
# --------------------------------------------------------------------------- #

# Legal suffix variants -> canonical form. Longest-match-first ordering handled
# by sorting keys by length descending at lookup time.
LEGAL_SUFFIXES = {
    "private limited": "private limited",
    "pvt ltd": "private limited",
    "pvt. ltd.": "private limited",
    "pvt limited": "private limited",
    "private ltd": "private limited",
    "p ltd": "private limited",
    "p pvt ltd": "private limited",
    "p pvt": "private limited",
    "p pvt limited": "private limited",
    "limited": "limited",
    "ltd": "limited",
    "ltd.": "limited",
    "corporation": "corporation",
    "corp": "corporation",
    "corp.": "corporation",
    "incorporated": "incorporated",
    "inc": "incorporated",
    "inc.": "incorporated",
    "l.l.c.": "llc",
    "l.l.c": "llc",
    "llc": "llc",
    "llp": "llp",
    "l.l.p.": "llp",
}
# Sort suffix keys by word-count descending so multi-word suffixes match before
# their substrings (e.g. "private limited" before "limited").
_SUFFIX_KEYS_SORTED = sorted(LEGAL_SUFFIXES.keys(), key=lambda k: -len(k.split()))

# --------------------------------------------------------------------------- #
# Abbreviation tables — split into two kinds with different expansion rules
# --------------------------------------------------------------------------- #

# ROAD_TYPE_ABBREVIATIONS: safe to expand wherever the token appears, since
# these rarely collide with ordinary English/Hindi words.
ROAD_TYPE_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "hwy": "highway",
    "apt": "apartment",
    "bldg": "building",
    "twp": "township",
    "twnshp": "township",
    "mrg": "marg",
}

# REGION_ABBREVIATIONS: US/Indian state codes. These are DANGEROUS to expand
# as a blind per-token replacement, because several collide with ordinary
# words: "in" (Indiana / the preposition "in"), "or" (Oregon / "or"),
# "hi" (Hawaii / greeting "hi"), "ok" (Oklahoma / "ok"), "me" (Maine / "me"),
# "ka" (Karnataka / Hindi "of", as in the locality "Lakdi-Ka-Pool").
#
# To stay safe, these are ONLY expanded when a token is its own standalone
# comma-separated segment of the address (e.g. "..., TX" or "..., RJ, ..."),
# which is how state codes actually appear in this data. A word like "in"
# embedded inside a phrase ("Sb Road In Haveli") has no surrounding commas
# isolating it, so it is left untouched. See _expand_regions_by_segment().
REGION_ABBREVIATIONS = {
    # US state codes
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming",

    # Indian state codes that don't collide with common words in this data.
    # Deliberately excludes "ka" (Karnataka) and "up" (Uttar Pradesh), which
    # collide with ordinary Hindi/English words.
    "wb": "west bengal",
    "mh": "maharashtra",
    "rj": "rajasthan",
    "mp": "madhya pradesh",
    "tg": "telangana",
    "hp": "himachal pradesh",
    "gj": "gujarat",
    "pb": "punjab",
    "jh": "jharkhand",
}

# Backward-compat alias in case other modules import the old combined name.
ADDRESS_ABBREVIATIONS = ROAD_TYPE_ABBREVIATIONS


# Simple landmark pattern (extend as more patterns are found in EDA).
LANDMARK_PATTERN = re.compile(r"\bnear\s+[a-z0-9 .]+", re.IGNORECASE)

# PIN / postal code pattern: standalone run of 4-6 digits.
PIN_PATTERN = re.compile(r"\b\d{4,6}\b")

# Placeholder values meaning "no data" that show up as literal text in
# address fields (e.g. "N/A" as a component between two real commas).
# Matched and removed BEFORE punctuation stripping, since stripping first
# would turn "N/A" into "n a" and hide it as two meaningless tokens.
_PLACEHOLDER_PATTERN = re.compile(r"\bn\s*/\s*a\b", re.IGNORECASE)

# Any standalone numeric token (for numeric_tokens extraction).
NUMERIC_TOKEN_PATTERN = re.compile(r"\b\d+\b")

# Non-Latin script detector: any character outside basic Latin + Latin-1 supplement
# + common punctuation/digits/whitespace ranges.
_NON_LATIN_PATTERN = re.compile(
    r"[^\u0000-\u024F\u2000-\u206F\s]"
)


# --------------------------------------------------------------------------- #
# Shared low-level helpers
# --------------------------------------------------------------------------- #

def _basic_clean(raw):
    """Unicode-normalize, lowercase, collapse whitespace. Returns '' for None/NaN."""
    if raw is None:
        return ""
    text = str(raw)
    if text.strip().lower() in ("", "nan", "none"):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[.,;:!?()\[\]{}'\"/\\|_#@*+=~`^-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_non_latin(raw):
    if raw is None:
        return False
    return bool(_NON_LATIN_PATTERN.search(str(raw)))


def _split_tokens(text):
    """Split on whitespace, drop empties. No dedup here — dedup happens
    AFTER abbreviation expansion, so multi-word expansions can't sneak
    duplicate words past the dedup step."""
    return [t for t in text.split(" ") if t]


def _dedupe_tokens(tokens):
    """Collapse immediate repeats, then dedupe overall, preserving first-seen order."""
    collapsed = []
    for t in tokens:
        if not collapsed or collapsed[-1] != t:
            collapsed.append(t)
    seen = set()
    out = []
    for t in collapsed:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _tokenize_dedup(text):
    """Convenience wrapper: split + dedupe, for text that has no further
    abbreviation expansion coming (used for names)."""
    return _dedupe_tokens(_split_tokens(text))


def _extract_suffix(clean_text):
    """Find a legal suffix at the end of clean_text. Returns (core_text, canonical_suffix_or_None)."""
    for suf in _SUFFIX_KEYS_SORTED:
        pattern = r"\b" + re.escape(suf) + r"\b$"
        if re.search(pattern, clean_text):
            core = re.sub(pattern, "", clean_text).strip()
            core = re.sub(r"\s+", " ", core)
            return core, LEGAL_SUFFIXES[suf]
    return clean_text, None


def _expand_regions_by_segment(raw_text):
    """
    Expand US/Indian state codes ONLY when a code is its own standalone
    comma-separated segment (e.g. "..., TX" or "..., RJ, ..."). This avoids
    treating ordinary words like "in", "or", "hi", "ka" as state codes when
    they appear embedded in a sentence with no comma isolating them.

    Returns a single cleaned string (segments rejoined with spaces).
    """
    segments = raw_text.split(",")
    processed = []
    for seg in segments:
        seg_clean = _basic_clean(seg)
        seg_tokens = seg_clean.split(" ") if seg_clean else []
        if len(seg_tokens) == 1 and seg_tokens[0] in REGION_ABBREVIATIONS:
            processed.append(REGION_ABBREVIATIONS[seg_tokens[0]])
        elif seg_clean:
            processed.append(seg_clean)
    return " ".join(processed)


def _expand_road_type_tokens(tokens):
    return [ROAD_TYPE_ABBREVIATIONS.get(t, t) for t in tokens]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def normalize_name(raw):
    """
    Normalize a business_name value.

    Returns a dict with keys:
        name_raw, name_clean, name_core, legal_suffix,
        name_tokens, name_tokens_sorted, is_non_latin, name_length
    """
    is_non_latin = _has_non_latin(raw)
    clean = _basic_clean(raw)
    core, suffix = _extract_suffix(clean)

    tokens_full = _tokenize_dedup(clean)
    tokens_sorted_str = " ".join(sorted(tokens_full))

    return {
        "name_raw": "" if raw is None else str(raw),
        "name_clean": clean,
        "name_core": core,
        "legal_suffix": suffix,          # None if not found
        "name_tokens": tokens_full,       # list[str]
        "name_tokens_sorted": tokens_sorted_str,
        "is_non_latin": is_non_latin,
        "name_length": len(clean),
    }


def normalize_address(raw):
    """
    Normalize a business_address value.

    Returns a dict with keys:
        address_raw, address_clean, address_tokens, address_tokens_sorted,
        pin_code, numeric_tokens, landmark_phrase, is_missing,
        is_non_latin, address_length
    """
    raw_str = "" if raw is None else str(raw)
    is_missing = raw_str.strip() == "" or raw_str.strip().lower() in ("nan", "none")
    is_non_latin = _has_non_latin(raw)

    if is_missing:
        return {
            "address_raw": raw_str,
            "address_clean": "",
            "address_tokens": [],
            "address_tokens_sorted": "",
            "pin_code": None,
            "numeric_tokens": [],
            "landmark_phrase": None,
            "is_missing": True,
            "is_non_latin": False,
            "address_length": 0,
        }

    # Extract landmark and PIN from the ORIGINAL text before heavy cleaning,
    # since punctuation stripping can mangle patterns like "near sbi atm".
    landmark_match = LANDMARK_PATTERN.search(raw_str.lower())
    landmark_phrase = landmark_match.group(0).strip() if landmark_match else None

    numeric_tokens_all = NUMERIC_TOKEN_PATTERN.findall(raw_str)
    pin_code = None
    for tok in numeric_tokens_all:
        if 4 <= len(tok) <= 6:
            pin_code = tok
            break

    raw_no_placeholder = _PLACEHOLDER_PATTERN.sub(" ", raw_str)

    # Region codes (state abbreviations) expanded ONLY on isolated comma
    # segments — see _expand_regions_by_segment for why this matters.
    region_expanded = _expand_regions_by_segment(raw_no_placeholder)

    # Now tokenize, expand road-type abbreviations (safe at any position),
    # and dedupe LAST so multi-word region expansions can't leave duplicates.
    raw_tokens = _split_tokens(region_expanded)
    road_expanded_tokens = _expand_road_type_tokens(raw_tokens)
    tokens = _dedupe_tokens(road_expanded_tokens)

    clean_expanded = " ".join(tokens)
    tokens_sorted_str = " ".join(sorted(tokens))

    return {
        "address_raw": raw_str,
        "address_clean": clean_expanded,
        "address_tokens": tokens,
        "address_tokens_sorted": tokens_sorted_str,
        "pin_code": pin_code,
        "numeric_tokens": numeric_tokens_all,
        "landmark_phrase": landmark_phrase,
        "is_missing": False,
        "is_non_latin": is_non_latin,
        "address_length": len(clean_expanded),
    }
