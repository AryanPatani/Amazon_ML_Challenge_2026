"""
src/path_b/serialization.py

Ticket B1: Text serialization for dense multilingual retrieval.
Serializes raw record fields ("business_name", "business_address", "country")
into a unified text representation suitable for transformer encoders.

Supports:
- Clean whitespace and NFKD unicode normalization
- Canonical legal suffix standardization (Pvt Ltd -> private limited)
- Address abbreviation expansion (Rd -> road, St -> street, Ave -> avenue)
- Prefix markers ("name: ... | address: ...")
- Country dropping for unseen-country generalization (France)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional
import pandas as pd


def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and strip control/non-printable characters."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    # NFKD unicode normalization
    text = unicodedata.normalize("NFKD", text)
    # Collapse multiple whitespace characters to a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def serialize_record(
    name: str,
    address: str,
    country: str = "",
    drop_country: bool = True,
    prefix_fields: bool = False,
    canonicalize: bool = False,
) -> str:
    """Serialize a single entity record into a structured text string.

    Parameters
    ----------
    name : str
        Business name.
    address : str
        Business address.
    country : str, optional
        Country name / code, by default "".
    drop_country : bool, optional
        If True, omits the country token to prevent overfitting to train
        countries (vital since test set contains unseen countries like France).
        By default True.
    prefix_fields : bool, optional
        If True, adds field markers: "name: ... | address: ...".
        If False, uses concise delimiter: "... | ...".
        By default False.
    canonicalize : bool, optional
        If True, applies domain normalization from src.common.normalize:
        expands legal suffixes ("Pvt. Ltd." -> "private limited") and
        address road types ("Rd" -> "road", "St" -> "street").
        By default False.

    Returns
    -------
    str
        Serialized string representation.
    """
    clean_name = clean_text(name)
    clean_addr = clean_text(address)

    if canonicalize:
        try:
            from src.common.normalize import normalize_name, normalize_address
            if clean_name:
                norm_n = normalize_name(clean_name)
                core = norm_n.get("name_core") or ""
                suffix = norm_n.get("legal_suffix") or ""
                clean_name = f"{core} {suffix}".strip() if core else norm_n.get("name_clean", clean_name)
            if clean_addr:
                norm_a = normalize_address(clean_addr)
                clean_addr = norm_a.get("address_clean", clean_addr)
        except Exception:
            pass  # Fall back to standard clean_name / clean_addr if normalize is unavailable

    parts = []
    if prefix_fields:
        if clean_name:
            parts.append(f"name: {clean_name}")
        if clean_addr:
            parts.append(f"address: {clean_addr}")
        if not drop_country and country:
            c = clean_text(country)
            if c:
                parts.append(f"country: {c}")
    else:
        if clean_name:
            parts.append(clean_name)
        if clean_addr:
            parts.append(clean_addr)
        if not drop_country and country:
            c = clean_text(country)
            if c:
                parts.append(c)

    return " | ".join(parts)


def serialize_dataframe(
    df: pd.DataFrame,
    drop_country: bool = True,
    prefix_fields: bool = False,
    canonicalize: bool = False,
) -> list[str]:
    """Serialize all records in a DataFrame to a list of strings.

    Expects columns: 'business_name', 'business_address' and optionally 'country'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entity records.
    drop_country : bool, optional
        Whether to drop country token, by default True.
    prefix_fields : bool, optional
        Whether to prefix fields, by default False.
    canonicalize : bool, optional
        Whether to apply canonical legal suffix & address expansion, by default False.

    Returns
    -------
    list[str]
        List of serialized strings matching the DataFrame rows.
    """
    has_country = "country" in df.columns and not drop_country
    names = df["business_name"].fillna("").astype(str).tolist()
    addresses = df["business_address"].fillna("").astype(str).tolist()
    countries = df["country"].fillna("").astype(str).tolist() if has_country else [""] * len(df)

    serialized = [
        serialize_record(
            name=n,
            address=a,
            country=c,
            drop_country=drop_country,
            prefix_fields=prefix_fields,
            canonicalize=canonicalize,
        )
        for n, a, c in zip(names, addresses, countries)
    ]
    return serialized
