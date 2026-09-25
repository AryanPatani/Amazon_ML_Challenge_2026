"""
test_normalize.py

Quick sanity tests for normalize.py, using real examples pulled from the
EDA report (train_ground_truth matched groups) plus edge cases.

Run:
    python3 test_normalize.py

No test framework needed — just run it and read the PASS/FAIL output.
If you have pytest installed, this also works with:
    pytest test_normalize.py -v
"""

from normalize import normalize_name, normalize_address

failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


def show(label, raw, result):
    print(f"\n--- {label} ---")
    print(f"  raw: {raw!r}")
    for k, v in result.items():
        print(f"    {k}: {v!r}")


# --------------------------------------------------------------------------- #
# NAME TESTS
# --------------------------------------------------------------------------- #

print("=" * 70)
print("NAME NORMALIZATION TESTS")
print("=" * 70)

r = normalize_name("Standard Automation Consultants Inc")
show("basic + suffix", "Standard Automation Consultants Inc", r)
check("suffix canonicalized to 'incorporated'", r["legal_suffix"] == "incorporated")
check("core name has suffix removed", r["name_core"] == "standard automation consultants")

r2 = normalize_name("standard automation consultants inc")
check("case-insensitive: same clean output as mixed case",
      r2["name_clean"] == r["name_clean"])

r3 = normalize_name("Modern Infra Private Limited")
r4 = normalize_name("Modern Private Infra Limited")  # word order swapped
show("word order A", "Modern Infra Private Limited", r3)
show("word order B", "Modern Private Infra Limited", r4)
check("word-order swap: sorted tokens match",
      r3["name_tokens_sorted"] == r4["name_tokens_sorted"])

r5 = normalize_name("ASSET BUILDING BUILDING INITIATIVE LLC")  # duplicated word
show("duplicated word", "ASSET BUILDING BUILDING INITIATIVE LLC", r5)
check("immediate duplicate token collapsed",
      r5["name_tokens"].count("building") == 1)

r6 = normalize_name("Rajdhani-(p) Pvt")
show("punctuation + suffix variant", "Rajdhani-(p) Pvt", r6)
check("suffix detected for '(p) pvt' style", r6["legal_suffix"] is not None)

r7 = normalize_name("Maure Williams Colombier & Co")
show("ampersand expansion", "Maure Williams Colombier & Co", r7)
check("'&' expanded to 'and'", "and" in r7["name_clean"])

r8 = normalize_name("ಶಕ್ತಿ ಕೇರ್")  # Kannada script
show("non-Latin script", "ಶಕ್ತಿ ಕೇರ್", r8)
check("non-Latin flag set True", r8["is_non_latin"] is True)

r9 = normalize_name(None)
show("None input", None, r9)
check("None input does not crash, returns empty clean", r9["name_clean"] == "")

r10 = normalize_name("")
check("empty string input handled", r10["name_clean"] == "")

# idempotency check
r11 = normalize_name("Standard Automation Consultants Inc")
r12 = normalize_name(r11["name_clean"])
check("idempotent: re-normalizing name_clean doesn't change it further",
      r12["name_clean"] == r11["name_clean"])


# --------------------------------------------------------------------------- #
# ADDRESS TESTS
# --------------------------------------------------------------------------- #

print("\n" + "=" * 70)
print("ADDRESS NORMALIZATION TESTS")
print("=" * 70)

a1 = normalize_address("85 Wayne Avenue, Ticonderoga, NY")
show("basic address", "85 Wayne Avenue, Ticonderoga, NY", a1)
check("state abbreviation NY expanded", "new york" in a1["address_clean"])

a2 = normalize_address("671 442, SWEETWATER, TX")
show("numeric + state abbrev", "671 442, SWEETWATER, TX", a2)
check("numeric tokens extracted", "671" in a2["numeric_tokens"] and "442" in a2["numeric_tokens"])
check("texas expanded", "texas" in a2["address_clean"])

a3 = normalize_address("Near SBI ATM, MG Road, Bangalore")
show("landmark phrase", "Near SBI ATM, MG Road, Bangalore", a3)
check("landmark phrase detected", a3["landmark_phrase"] is not None and "near" in a3["landmark_phrase"])

a4 = normalize_address("")
show("empty address", "", a4)
check("empty address flagged is_missing=True", a4["is_missing"] is True)
check("empty address has empty tokens", a4["address_tokens"] == [])

a5 = normalize_address(None)
check("None address handled without crash", a5["is_missing"] is True)

a6 = normalize_address("Kolkata, Calcutta, 31 B B D Bagh S, West Bengal")
a7 = normalize_address("31 B B D Bagh S, Calcutta, WB, Kolkata")  # reordered + abbreviation
show("reordered address A", "Kolkata, Calcutta, 31 B B D Bagh S, West Bengal", a6)
show("reordered address B", "31 B B D Bagh S, Calcutta, WB, Kolkata", a7)
check("WB abbreviation expands to match 'west bengal'",
      "west bengal" in a7["address_clean"])
check("reordered addresses: sorted tokens overlap heavily",
      len(set(a6["address_tokens"]) & set(a7["address_tokens"])) >= 5)

a8 = normalize_address("10A, Kasturba Gandhi Marg, New Delhi, Delhi 110001")
show("PIN code extraction", "10A, Kasturba Gandhi Marg, New Delhi, Delhi 110001", a8)
check("PIN code extracted", a8["pin_code"] == "110001")

a9 = normalize_address("77, 1St Floor, Nrupatunga Main Road, Mysore, ಕರ್ನಾಟಕ")
show("mixed script address", "77, 1St Floor, Nrupatunga Main Road, Mysore, ಕರ್ನಾಟಕ", a9)
check("non-Latin flag set True for mixed-script address", a9["is_non_latin"] is True)

a12 = normalize_address("Lakdi- Ka, -Pool, Hyderabad, Telangana")
show("Hindi word 'ka' should NOT become Karnataka", "Lakdi- Ka, -Pool, Hyderabad, Telangana", a12)
check("'ka' is not wrongly expanded to karnataka", "karnataka" not in a12["address_clean"])

a13 = normalize_address("##19821 WHEELWRIGHT DR, MONTGOMERY VILLAGE, MD")
show("stray hash symbols", "##19821 WHEELWRIGHT DR, MONTGOMERY VILLAGE, MD", a13)
check("'#' characters stripped", "#" not in a13["address_clean"])

a14 = normalize_address("CALUMET CITY, 351 HOXIE AVE, N/A, IL")
show("N/A placeholder", "CALUMET CITY, 351 HOXIE AVE, N/A, IL", a14)
check("'n/a' placeholder removed, not split into 'n' 'a' tokens",
      "n" not in a14["address_tokens"] and "a" not in a14["address_tokens"])

a15 = normalize_address("Jaipur, RJ")
a16 = normalize_address("Jaipur, Rajasthan")
show("state abbrev RJ", "Jaipur, RJ", a15)
show("state full Rajasthan", "Jaipur, Rajasthan", a16)
check("RJ expands to match 'rajasthan'", "rajasthan" in a15["address_clean"])
check("RJ and Rajasthan versions share all tokens",
      set(a15["address_tokens"]) == set(a16["address_tokens"]))

a17 = normalize_address("Opp Ratna Hospital Sb Road In Haveli, Pune, Maharashtra")
show("'in' preposition should NOT become Indiana", "Opp Ratna Hospital Sb Road In Haveli, Pune, Maharashtra", a17)
check("'in' is not wrongly expanded to indiana", "indiana" not in a17["address_clean"])
check("'in' remains as a plain token", "in" in a17["address_tokens"])

a18 = normalize_address("1St Floor, Sehore, Madhya Pradesh, MP")
show("MP expansion should not duplicate existing 'madhya pradesh'", "1St Floor, Sehore, Madhya Pradesh, MP", a18)
check("no duplicate 'madhya'/'pradesh' tokens after MP expansion",
      a18["address_tokens"].count("madhya") == 1 and a18["address_tokens"].count("pradesh") == 1)

a19 = normalize_address("Calumet City, IL")
show("standalone state code IL still expands correctly", "Calumet City, IL", a19)
check("IL still expands to illinois when isolated", "illinois" in a19["address_clean"])

# idempotency check
a10 = normalize_address("1671 442, Sweetwater, TX")
a11 = normalize_address(a10["address_clean"])
check("idempotent: re-normalizing address_clean doesn't change it further",
      a11["address_clean"] == a10["address_clean"])


# --------------------------------------------------------------------------- #
# SUMMARY
# --------------------------------------------------------------------------- #

print("\n" + "=" * 70)
if failures:
    print(f"SUMMARY: {len(failures)} FAILURE(S)")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("SUMMARY: ALL TESTS PASSED")
