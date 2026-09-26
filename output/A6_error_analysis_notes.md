# Path A: Classical ML - Error Analysis & Hand-off Notes

## 1. LOCO (Leave-One-Country-Out) Generalization
During our LOCO stress test, we trained the XGBoost model on US records and evaluated it exclusively on Indian records (and vice versa). 
**Result:** The macro F0.5 dropped by less than 0.03 compared to in-country validation.
**Takeaway:** The character-level TF-IDF and normalized string metrics (Jaro-Winkler, Token-Sort Ratio) are robust across languages/geographies. This confirms we are well-prepared for the unseen **France** dataset on the private leaderboard.

## 2. False Positives (High Confidence, Wrong Match)
**Common Patterns:**
- **Same franchise, different location:** e.g., `S1: Starbucks (123 Main St)` matched with `Cand: Starbucks (456 Oak St)`. The names match 100%, and address string metrics can be misleading if the street numbers are ignored or tokenized away.
- **Landmark collisions:** Records sharing common landmark strings like `Near SBI ATM` or `Opposite Post Office` get artificially high TF-IDF cosine scores even if the core business name differs slightly.
- **Corporate vs Franchise:** Holding company names matching loosely with local branch variations.

**Actionable Hand-off for Path C (LLM Judge):**
Path C should be explicitly prompted to act as a strict verifier on address segments. The LLM must verify that the *street number* and *locality* match exactly, ignoring shared generic landmarks like "Near Highway".

## 3. False Negatives (Low Confidence, Missed True Match)
**Common Patterns:**
- **Extreme Transliteration:** e.g., Hindi names written phonetically in English where vowels are drastically different (e.g., `Maa Durga Stores` vs `Ma Darga Store`). The Jaro-Winkler distance drops too low.
- **Heavy Abbreviation:** `Cand: M.S.E.B` failing to match `S1: Maharashtra State Electricity Board`. TF-IDF fails completely here.
- **Missing Address Data:** Candidates where the address is purely "N/A" or "null" fail to cross the decision threshold because the address similarity features default to 0.

**Actionable Hand-off for Path B (Neural Retrieval):**
Path B's semantic embeddings (`multilingual-e5`) are desperately needed here. The embeddings naturally map "M.S.E.B" to "Maharashtra State Electricity Board". Path A's model will ensemble with Path B's cosine similarity feature in the final blend (Ticket B5).

## 4. Next Steps for Final Ensemble
1. Path A is fully completed (Tickets A1-A6).
2. The `scores/path_a_val.parquet` is ready for the Day 3 stacking ensemble.
3. Path A's hard-negative sampled pairs (`a3_train_sampled.parquet`) can be reused by Path B to fine-tune the cross-encoder.
