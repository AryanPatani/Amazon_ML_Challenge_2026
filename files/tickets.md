# Amazon ML Challenge 2026 — Business Entity Resolution
## PRD + 4 Parallel Paths + Tickets

**Window:** 25 Sep 12:00 AM → 27 Sep 11:59 PM IST (72 h). Only the team leader can upload to the leaderboard.
**Deliverables:** `matching_results.tsv` (scored), `candidate_pairs.tsv` (blocking output), zipped code + README + requirements.txt, filled `Documentation_template.md`, 1–2 page approach document.

---

## 1. Problem in one paragraph

Three sources hold noisy records about the same real-world businesses (name, address, country). Source 1 is the clean, deduplicated reference. For every Source 1 entity we must output the list of Source 2 and Source 3 records that are the same business (0, 1 or many). Score = F0.5 per S1 entity, averaged (macro). Singletons (no true match) score 1.0 if we output an empty list and 0.0 if we output anything, so "when in doubt, leave it out".

## 2. Goals and non-goals

**Goals**
- Maximise macro F0.5 on the private leaderboard.
- Generalise to **France**, which is absent from training.
- Produce a valid, reproducible package (validator passes, candidates ⊇ matches).
- Comply with rules: no external lookups / geocoding APIs; final model MIT/Apache-2.0 and ≤ 8B parameters.

**Non-goals**
- Perfect recall. Precision is weighted 2x, so we trade recall for precision deliberately.
- Any external data enrichment (disqualification risk).

## 3. Key facts and design implications

| Fact | Implication |
|---|---|
| S1 is deduplicated | A given S2/S3 record should match **at most one** S1 entity. Verify on train; if true, enforce it as a global assignment constraint (big precision win). |
| Many matches per S1 allowed | Output is a set, not top-1. Need a per-pair decision, not just argmax. |
| Singletons count fully | Need a calibrated "no match" threshold; a S1 entity with only weak candidates must return empty. |
| Macro over S1 entities | Every entity weighs the same, regardless of country or number of matches. |
| France unseen in train | No country one-hot, no hard-coded country lists. Prefer language-agnostic, character-level and multilingual features. Validate with leave-one-country-out. |
| candidate_pairs.tsv = exact set fed to final model | Record the candidate set at the last stage right before model inference. Matched IDs must be a subset. |
| Noise types | Abbreviations, legal suffixes, typos, transliteration, word-order, missing PIN/state, landmark addresses, reordered components. |
| Tab-separated files | `pd.read_csv(..., sep="\t")` always. |

## 4. Shared contract (so paths can merge later)

Every path must write scored pairs in one common format so we can ensemble and threshold on Day 3.

- **Pair file:** `scores/<path>_{val,test}.parquet` with columns `s1_id, cand_id, score` (score in [0,1], higher = more likely match).
- **Common validation split:** made once in ticket 0.1, grouped by S1 entity, stratified by country and by singleton/non-singleton. Saved as `splits/val_s1_ids.txt`. Nobody re-splits.
- **Extra stress split:** leave-one-country-out (train on US → validate on India and vice versa) to simulate France.
- **Metric:** one shared function `eval/f05.py` computing macro F0.5 including singletons.
- **Repo layout:** `src/common/`, `src/path_a/`, `src/path_b/`, `src/path_c/`, `src/path_d/`, `scores/`, `output/`.

## 5. Shared foundation (first ~2 hours, split across all 4)

| ID | Ticket | Owner | Done when |
|---|---|---|---|
| 0.1 | Data loader (tsv, explicit tab sep), stats notebook: sizes, matches per S1, fraction singletons, S2-vs-S3 split, check whether an S2/S3 record ever matches >1 S1 | Member 1 | Stats posted to team; "at most one S1" claim confirmed or refuted |
| 0.2 | `eval/f05.py` macro F0.5 with singleton handling + fixed val split + leave-one-country-out split | Member 2 | Unit-tested on the worked example in the PS (0.714) |
| 0.3 | Text normaliser v0: lowercase, unicode NFKD, strip punctuation, expand/contract legal suffixes (Pvt/Private, Ltd/Limited, Corp/Corporation), `&`→`and`, address abbreviations (Rd/Road, St/Street) | Member 3 | Shared `normalize_name()`, `normalize_address()` in `src/common/` |
| 0.4 | Output writers + wrapper around `utils/validate_submission.py`; writes both TSVs from a scored pair file + threshold | Member 4 | Dummy submission (all empty lists) passes validator |

After this, everyone splits into their own path.

---

## 6. Path A — Classical ML: blocking + engineered features + gradient boosting

**Idea:** Strong, fast, interpretable baseline. Likely the backbone of the final system.
**Why it generalises to France:** char-level similarity features are language-agnostic.

| ID | Ticket | Est. | Acceptance |
|---|---|---|---|
| A1 | **Blocking:** union of (a) char n-gram TF-IDF top-K by name, (b) token-overlap on rare name tokens, (c) address token/PIN/postcode blocks, (d) phonetic keys (Double Metaphone / Soundex on name tokens). Measure recall ceiling vs candidates per S1 | 3 h | Candidate recall ≥ 98% on train; avg candidates per S1 reported |
| A2 | **Pair features:** Jaro-Winkler, Levenshtein ratio, token-set/sort ratio, Jaccard on tokens and char 3-grams, TF-IDF cosine (name, address), first-token match, acronym match, numeric-token overlap (building no., PIN), PIN/state agreement flags, missing-field flags, length ratios | 4 h | Feature table for train/val/test pairs in parquet |
| A3 | **Negative sampling:** hard negatives from blocking (not random) so the classifier sees realistic confusers | 1 h | Class balance and pair counts documented |
| A4 | **LightGBM/XGBoost classifier** with grouped CV by S1 entity; no country feature | 3 h | Val macro F0.5 reported; scores written in shared format |
| A5 | **Decision layer:** global threshold tuned for F0.5 + one-to-one assignment (each S2/S3 record to best S1 only) + singleton rule (relative gap between top-1 and top-2, minimum absolute score) | 2 h | Val F0.5 gain from assignment and singleton logic shown separately |
| A6 | Leave-one-country-out evaluation and error analysis (top 50 false positives / false negatives) | 2 h | Written notes fed to other paths |

---

## 7. Path B — Neural retrieval + cross-encoder re-ranking

**Idea:** Semantic embeddings catch transliterations, word-order changes and abbreviations that string metrics miss; a fine-tuned cross-encoder gives the precise pair decision.
**Model constraint:** MIT/Apache-2.0, ≤ 8B. Candidates: `multilingual-e5-base/large` (MIT), `bge-m3` (MIT), `all-MiniLM` family (Apache-2.0), XLM-R (MIT). Verify each licence on its model card before use. Multilingual models matter for France.

| ID | Ticket | Est. | Acceptance |
|---|---|---|---|
| B1 | **Text serialisation:** `"name | address"` strings (with and without country token dropped) and embed S1/S2/S3 with an off-the-shelf multilingual encoder; FAISS/`numpy` top-K retrieval | 3 h | Recall@20/50 on train reported vs Path A blocking |
| B2 | **Fine-tune bi-encoder** with contrastive loss (MultipleNegativesRankingLoss) using train matches, hard negatives from B1 | 4 h | Recall@K improves over zero-shot |
| B3 | **Cross-encoder** (small multilingual encoder) fine-tuned on (S1, candidate) pairs, top-K from B2 as input | 5 h | Val AUC/F0.5 reported, scores in shared format |
| B4 | **Augmentation for robustness:** synthetic noise (drop PIN, swap word order, abbreviate, transliterate variants, landmark strings) to simulate unseen-country noise | 2 h | Ablation: with vs without augmentation on leave-one-country-out |
| B5 | Export retrieval scores and embedding-cosine as extra features for Path A's model | 1 h | `scores/path_b_*.parquet` available to A and D |

Compute: use AWS credits ($200, +$100 for top 500 at 48 h) for GPU training; a single small GPU is enough.

---

## 8. Path C — LLM as verifier / judge on hard pairs

**Idea:** Use an open LLM ≤ 8B (e.g. Qwen 7B/8B class, Apache-2.0 — check the exact model card) only on the **uncertain band** of pairs from other paths. Good at reasoning about landmark addresses, transliteration, DBA names.
**Cost control:** never run on all candidates; only on pairs whose Path A/B score is in the uncertain range.

| ID | Ticket | Est. | Acceptance |
|---|---|---|---|
| C1 | Set up inference (vLLM or HF transformers, 4-bit if needed) on AWS credits; measure throughput on a sample | 2 h | Pairs/second known; fits time budget for test set uncertain band |
| C2 | **Zero-shot prompt** returning match/no-match + confidence (logprob of "yes"); few-shot examples of each noise type | 3 h | Val F0.5 on hard pairs vs Path A score alone |
| C3 | **LoRA fine-tune** on train pairs (prompt → yes/no), balanced with hard negatives | 5 h | Beats zero-shot on val; licence of base model documented |
| C4 | Calibrate LLM probability (isotonic/Platt) so it can be blended with other scores | 2 h | Reliability curve; scores in shared format for uncertain band |
| C5 | Failure analysis: where LLM overrides classical model and is right or wrong | 2 h | Short report for the Day 3 ensemble |

Watch out: no external API calls (rule says local model only), and keep the final model ≤ 8B and MIT/Apache-2.0.

---

## 9. Path D — Unsupervised / collective ER + graph clustering + generalisation

**Idea:** Robustness path. Uses the structure of the problem (S1 deduplicated, transitive matches between S2 and S3) instead of just pairwise scores, and does not depend on country-specific patterns.

| ID | Ticket | Est. | Acceptance |
|---|---|---|---|
| D1 | **Probabilistic baseline (Fellegi–Sunter / EM style)** on comparison vectors (name sim bins, address sim bins, PIN match) — needs no country label, useful as a sanity baseline on France-like data | 3 h | Val F0.5 reported; scores in shared format |
| D2 | **Graph construction:** nodes = S1/S2/S3 records, edges = scored pairs (from A/B). Add S2–S3 edges (same business appearing in both) | 3 h | Graph built for val and test |
| D3 | **Collective decision:** connected components / correlation clustering with constraint "one S1 per component"; use S2–S3 agreement to boost or veto weak S1–S2 / S1–S3 edges | 4 h | Val F0.5 gain over independent thresholding |
| D4 | **Singleton detector:** dedicated model/heuristic deciding whether an S1 entity has any match (margin between best score and background distribution) | 3 h | Singleton precision/recall reported |
| D5 | **France proxy test:** leave-one-country-out, plus stress test where addresses are stripped of PIN/state and names transliterated | 2 h | Table of degradation per path |
| D6 | Build `candidate_pairs.tsv` generation + reproducibility README + requirements.txt + Documentation_template.md skeleton | 3 h | Package builds from scratch with one command |

---

## 10. Integration plan (Day 3)

| Time | Task |
|---|---|
| Day 1 (today) | Foundation (Section 5), then each path reaches its first scored pair file. Upload a Path A baseline to the leaderboard early to confirm the format is accepted. |
| Day 2 | Paths finish core tickets. Share error analyses. Path B scores → Path A features. At the 48 h mark, top-500 teams get extra $100 credits. |
| Day 3 morning | **Stacking:** logistic regression / small LightGBM on val over scores from A, B, C (uncertain band only), D. Fit only on val; avoid overfitting the public leaderboard. |
| Day 3 midday | Apply D's assignment + singleton logic on the blended score. Tune threshold on val for F0.5 (bias towards precision). |
| Day 3 afternoon | Generate final `matching_results.tsv` and `candidate_pairs.tsv` (last stage before model), run validator, upload. |
| Day 3 evening | Assemble zip, finish documentation and 1–2 page approach doc. Leave buffer before 11:59 PM IST. |

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfitting public leaderboard | Trust grouped val + leave-one-country-out more than leaderboard deltas |
| France collapses precision | No country features, multilingual encoder, noise augmentation, conservative threshold |
| Candidate set ≠ what model saw | Single function that returns the final candidate set and feeds inference; write `candidate_pairs.tsv` from that same object |
| Licence violation | Keep a `MODELS.md` listing each model, size, licence, source URL |
| Compute overrun | Path C only on uncertain band; cache embeddings; monitor AWS credit spend |
| Format rejection | Run validator after every export; one row per S1 entity, no duplicates, only S2/S3 IDs that exist in test |
| Fair-play breach | No geocoding or external lookups anywhere in code; review imports before submission |

## 12. Success metrics

- Val macro F0.5 (including singletons) — primary.
- Candidate recall ceiling ≥ 98% with a large reduction ratio.
- Leave-one-country-out F0.5 within a few points of in-country F0.5.
- Validator: PASS on both files.
- Reproducible run from README on a clean environment.

## 13. Suggested owner mapping

| Member | Path |
|---|---|
| 1 | A — Classical ML (also final integrator) |
| 2 | B — Neural retrieval + cross-encoder |
| 3 | C — LLM verifier |
| 4 | D — Graph / collective + packaging |

Swap based on strengths; the leader (who uploads to the leaderboard) should ideally own the integration.
