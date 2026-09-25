# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** Faith
**Team Members:** Ali Ansari (Team Leader), Raghav Malani
**Submission Date:** {{SUBMISSION_DATE}}

---

## 1. Executive Summary

We resolve business entities across three noisy sources with a blocking + classifier
pipeline: normalise text, retrieve a short candidate list per Source 1 entity with
hashed TF-IDF cosine similarity (blocking), score every candidate pair with a
LightGBM model trained on similarity features, give each Source 2/3 record to at
most one Source 1 entity, and keep the pairs whose probability clears a cutoff
chosen by measuring the exact macro F0.5 metric on held-out entities. The pipeline is country-agnostic by construction (no
country feature is ever passed to the model), which is what lets it handle France in
the test set despite having zero French training examples. On a held-out validation
split of training S1 entities, this reaches a macro F0.5 of 0.9688
(US 0.9793, India 0.9529).

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training data shaped every downstream design decision.

**Scale.** Training has 2.21M Source 1 entities, 5.03M Source 2, and 5.29M Source 3
records. The test set has 1.73M Source 1 entities split by country as India 810k,
US 663k, France 259k, plus 4.89M Source 2 and 5.08M Source 3 records. Comparing
every Source 1 entity against every Source 2/3 record is computationally
impossible; candidate generation is not optional.

**Match structure.** Only 5.6% of training Source 1 entities are singletons (no
match at all); the rest have on average about 3.5 matches, up to 5 from Source 2 and
up to 6 from Source 3. Every Source 2/3 record belongs to at most one Source 1
entity: across all 7.64M labelled links in training, multiplicity is exactly 1. This
directly justifies a one-owner-per-record assignment rule at decision time (see
Section 4). Matches never cross countries: 0 cross-country matches in a 200k-pair
sample. This means blocking and scoring can be partitioned by country with no loss
of true matches, which is also a large speed win.

**Hard negatives.** About 27% of Source 2/3 records (2.68M in training) match
nothing at all. Many of these are not random noise but deliberately planted decoys
sitting right next to a real Source 1 entity: same or near-identical name, address
one digit off. Examples found in training:
- `Hong Management | 20996 Fm 2556, Santa Rosa, TX` vs. decoy
  `Hong Management Inc | 20999 FM 2556` (near-identical address, extra legal suffix)
- `Luciani and Markel Hospitality LLC | 3670 Jeanna Dr` vs. decoy
  `Luciani and Markel Excavation LLC | 3675 Jeanna Dr` (same firm name, different
  business line, nearby address)
- `Super Commodities Private Limited` vs. decoy `Super Commodities L.L.P.` at the
  same address (identical name, different legal form)

These decoys are why the model needs features that specifically separate "same
business, sloppy entry" from "different business, suspiciously similar entry": a
raw name-similarity score alone cannot tell them apart.

**True-match noise patterns** observed and specifically handled in normalisation
and features: dropped digits in numbers (6100 to 100), zero-padding (357 to 00357),
letter suffixes on house numbers (127 to 127-B), word swaps and synonym
replacements, typos and leetspeak (Br0thers, C0astal), Hindi/Kannada script names
(24% of India Source 2 names are non-Latin), websites used as a business name
(interfaithcenter.com), "X formerly known as Y" and "nee" aliasing, "(ID: 23374)"
tags, embedded phone numbers, "M/s" prefixes, reordered address components, literal
"NULL" tokens, US state name vs. two-letter code, Indian state names in native
script, and French-specific patterns: abbreviations (`R.`, `Av`, `Bd`, `N°`) and
legal forms (`SARL`, `SAS`, `EURL`, `SCI`).

**The France problem.** Training covers only US and India. Test adds France with
zero French training examples, and country must be treated as an open string label,
not a fixed `{US, India}` set. This ruled out any country-specific feature, lookup
table keyed by country, or per-country model. See Section 2.2 and the France
subsection under Section 4 for how the pipeline handles this.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage entity resolution): a
recall-oriented candidate generation stage followed by a precision-oriented pairwise
classifier and a metric-aware decision rule.

**Core Innovation:** Every design choice is driven directly by the competition
metric and by facts measured in the data, rather than by generic entity-resolution
defaults:
- Blocking runs in two directions (each Source 2/3 record retrieves its own best
  Source 1 candidates, in addition to the usual Source 1 to pool direction) because
  the one-owner structure of the data makes the reverse direction unusually precise.
- The decision rule is chosen by the metric, not by convention. After one-owner
  assignment, two rule families compete on held-out entities under the exact
  competition formula: a plain probability threshold (grid 0.20 to 0.95) and an
  expected-F0.5 top-k rule that scores every cutoff k, including k = 0 (predict
  singleton). The plain threshold at 0.725 won by a small margin, so the simpler
  rule ships.
- Validation reproduces test conditions: before measuring, the model scores every
  training pair of all 2.21M Source 1 entities, so the one-owner competition is as
  crowded as it will be on the test set.
- No feature ever encodes country directly, so the same trained model applies to
  France without modification.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used.** Not a single key but hashed TF-IDF vectors over word
unigrams and bigrams (`HashingVectorizer`, 2^24 buckets, no vocabulary held in RAM),
built separately for name text and address text and fit per country. Address terms
are prefixed so they can never collide with name terms in the combined vector. A
bigram like "12029 sheraton" (house number plus street) is rare enough to pin down
the right record cheaply; unigrams keep recall when the number itself is corrupted.
Blocking, like scoring, is done per country, which is safe because matches never
cross countries in the data.

Two retrieval directions are computed and unioned, both using sparse top-k cosine
similarity (`sparse_dot_topn`):
- **Reverse:** every Source 2/3 record retrieves its `k_rev` most similar Source 1
  records. Because each Source 2/3 record belongs to at most one Source 1 entity,
  this direction is unusually precise: the true owner lands at rank 1 for 95.3% of
  records and in the top 3 for 97.1%.
- **Forward:** every Source 1 record retrieves its `k_fwd` most similar records,
  separately per source (Source 2, Source 3). This recovers true owners that a
  generic-looking Source 2/3 record ranked low from the reverse direction.

Before the similarity search, terms that appear in more than `prune` (default
0.2%) of a country's records are dropped from the query side only. These very
common terms barely move a cosine score but dominate the cost of the sparse matrix
product, so pruning them is close to free in recall and large in speed.

**How we ensured true matches were not lost.** Several blocking configurations were
measured directly against known true pairs before settling on the final one:
- Forward-only retrieval (Source 1 to pool, top 10) reached only about 90% recall
  and was slow, because a badly-worded Source 2/3 record can rank far from its true
  Source 1 owner even though the reverse lookup would find it immediately.
- A name-only character-trigram pass was tried as a cheap first filter but was weak
  specifically in the reverse direction (rank-1 recall only about 38%) and was
  dropped rather than layered in.
- Adding word bigrams (hashed unigram+bigram TF-IDF) ran about 3.5x faster than
  word unigrams alone at the same recall: rare bigrams such as "12029 sheraton" let
  the search drop common unigrams far more aggressively. This is the final
  representation.
- **Final configuration:** reverse top-3 unioned with forward top-10 per source,
  query-side pruning at document frequency > 0.2%. Measured recall: 97.9% of true
  pairs survive on a US sample, at about 20 candidates per Source 1 entity. Recall
  on the full training run (all countries, this exact configuration):
  **96.65%** of true train pairs survive blocking, at **56.19M** candidate pairs for
  the 2.21M train Source 1 entities (25.5 candidates per Source 1 entity).

**Candidate pairs generated.** Total candidate pairs on the test set (the exact set
handed to the matching model, and what is written to `candidate_pairs.tsv`):
**48,053,510** (US 18.13M, India 22.31M, France 7.62M; 27.7 per Source 1 entity).

Each candidate pair also carries "context" features computed during blocking
(cosine similarity on the combined/name-only/address-only vectors, each candidate's
rank and score-gap among its own Source 1 entity's other candidates, and the
reverse gap among the Source 1 entities that considered this same Source 2/3
record). These become part of the classifier's feature set in Section 4.

---

## 4. Matching Model

### Features used

Every feature is a similarity between the Source 1 record and one candidate, never
a raw value and never a country flag, so the classifier transfers to France without
having seen any French training pairs.

| Group | What it captures | Representative features |
| --- | --- | --- |
| **Name** | How alike are the cleaned business names, their aliases, and their phonetic/no-space forms. Covers typos, transliteration, word-order swaps, and "formerly known as" aliasing. | `n_ratio`, `n_tset`, `n_tsort`, `n_partial` (edit-distance and token-set/sort similarity via RapidFuzz), `n_jw` (Jaro-Winkler), `n_first_eq`, `n_len_a`/`n_len_b`, `n_alias` (best match against either side's "aka" name), `n_nosp`/`n_nosp_partial` (space-stripped, catches website-as-name cases), `n_phon`/`n_phon_tset` (consonant-skeleton phonetic key, robust to transliteration), `nl_a`/`nl_b` (was the raw name non-Latin script) |
| **Legal** | Whether either side carries a legal-form tag (Ltd, LLC, SARL, ...) and whether the tags agree or actively conflict. A conflict (e.g. Hospitality LLC vs. Excavation LLC does not conflict, but Private Limited vs. LLP does) is a strong decoy signal. | `leg_a`, `leg_b`, `leg_same`, `leg_conflict` |
| **Address + numbers** | Fuzzy similarity of the cleaned address text, plus explicit house-number/PIN-number logic. This is the group specifically built to separate true noise (dropped digits, zero-padding) from decoys (same street, house number off by a handful). | `a_tsort`, `a_tset`, `a_partial`, `a_empty_b`; `num_first_eq`, `num_common`, `num_jacc`, `num_only_a`/`num_only_b`, `num_trunc` (one number is a truncation/suffix of the other, e.g. 6100 to 100), `num_near` (numbers differ by ≤20, the typical decoy gap), `num_logdiff`, `num_digits` |
| **TF-IDF context (from blocking)** | Where this pair stands relative to its competitors, from the blocking stage's similarity search. High context scores mean the pair also looks strong in aggregate word-overlap terms; a large gap to the best-ranked alternative means the pair is not just plausible but clearly the best option. | `cos_w`, `cos_nw`, `cos_aw` (combined/name-only/address-only cosine), `rank_rev`, `rank_fwd`, `unm_a`/`unm_b` (share of TF-IDF weight on words the other side lacks entirely, high on both sides for decoys with one differing distinctive word), `*_gap`/`*_rk`/`*_rgap` (distance and rank to this Source 1's best candidate, and to this candidate's best Source 1), `n_s1_for_i`, `n_high` |

Missing-data handling: when a side of a comparison is empty (no alias, no address
number), the corresponding feature is `NaN` rather than a manufactured "0 similarity"
value, and LightGBM handles missing values natively in its split-finding, so absence
of a field is not confused with active dissimilarity.

### Model architecture

**Model type:** LightGBM (`lightgbm` 4.7.0), a gradient-boosted decision tree
ensemble trained as a binary classifier ("is this Source 1/candidate pair the same
business"). Key hyperparameters: 127 leaves per tree, learning rate 0.08, feature
and bagging fraction 0.8 (regularisation against the huge number of near-duplicate
candidate pairs per entity), L2 regularisation 1.0, up to 2000 boosting rounds with
early stopping (patience 50) against a held-out validation set. LightGBM is
MIT-licensed and, as a tree ensemble rather than a parameter-counted neural network,
trivially satisfies the challenge's "MIT/Apache-2.0, ≤8B parameters" model
constraint.

Training uses candidate pairs from a sample of Source 1 entities (`--train-s1` +
`--valid-s1`, split three ways, see Validation protocol below). Top features by
share of gain on this run: `cos_w_rgap` 44.7%, `rank_rev` 14.5%,
`cos_w` 10.8%, `num_logdiff` 5.2%, `num_near` 2.6%, `a_tset` 2.6%, `cos_nw_rgap`
2.3%, `n_jw` 2.0%, `num_jacc` 1.4%, `n_phon` 1.4%. The single dominant feature,
`cos_w_rgap` (how far this Source 1 entity trails the candidate's best-matching
Source 1 entity in combined-vector cosine similarity), is a blocking-stage
"reverse competition" signal: it directly encodes the one-owner intuition from
Section 2.1 that a real match is usually the *best* claim on a Source 2/3 record,
not just *a* plausible one. `rank_rev` (this candidate's rank among the Source 1
entities that retrieved it in the reverse blocking direction) reinforces the same
idea. Name, address, and number features matter but individually carry much less
weight than these two competition-based context features.

### Validation protocol

Getting an honest validation number required more care than a single train/test
split. The sampled Source 1 entities (`--train-s1` 300,000 + `--valid-s1` 100,000,
fixed random seed) are split three ways: 300,000 fit the LightGBM model, 50,000
early-stop it (LightGBM's own `valid_sets`), and the remaining 50,000 are held back
untouched to choose the decision rule and report the score below.

Before scoring those final 50,000 entities, the trained model scores every
candidate pair for all 2.21M train Source 1 entities, not only the sampled ones,
and the one-owner assignment (Section "Decision rule" below) runs over that full
scored set. This matters because one-owner assignment is a competition between
Source 1 entities for a shared Source 2/3 record, and on the test set a sampled
entity's candidate competes against every other test Source 1 entity, sampled or
not. Scoring only the sampled 400,000 entities and running one-owner on that
subset would let a held-out entity keep a record that an unsampled entity would
actually have won on the full set, silently inflating the validation score. A first
run without this correction scored 0.9689 against 0.9688 with it: the bias was
small here, but the corrected number is the one reported in Section 5.

This is also why the model code separates the old single decision function into
two pieces: `matcher.one_owner()` runs once over the full scored set (all 2.21M
Source 1 entities), and `matcher.select(rule)` then applies a chosen decision rule
afterward, cheaply enough per Source 1 entity to try several candidate rules
without re-scoring anything.

### Decision rule

The trained model outputs a probability `p` per candidate pair. Two rules apply on
top of `p` to produce the final per-entity match lists, in this order:

**1. One-owner assignment.** Since the data guarantees every Source 2/3 record
belongs to at most one Source 1 entity, each candidate record is kept only for the
Source 1 entity that gave it the single highest probability across every Source 1
entity that considered it as a candidate; every other (weaker) claim on that record
is dropped before anything else happens.

**2. How many to keep per entity.** The competition metric, for a Source 1 entity
with `|truth|` true matches, predicting `k` matches of which `TP` are correct, is:

```
F_0.5 = 1.25 * TP / (0.25 * |truth| + k)
```

(algebraically identical to the standard `1.25*P*R/(0.25*P+R)` form with
`P = TP/k`, `R = TP/|truth|`; a Source 1 entity with `|truth| = 0` scores 1.0 if
`k = 0` and 0.0 for any `k > 0`.) Substituting the model's own probabilities as
unbiased estimates (sum of the top-k kept probabilities standing in for expected
`TP`, sum of all candidate probabilities for that entity standing in for expected
`|truth|`) turns this into a score computable for every candidate cutoff `k`,
including `k = 0` (predict singleton), without needing the true labels at
prediction time. The pipeline evaluates every `k` for every Source 1 entity and
keeps whichever `k` maximises this expected F0.5. A plain fixed probability
threshold is evaluated in parallel over a grid of cutoffs. Both rule families are
scored with the true competition formula on the held-out validation Source 1
entities, and whichever rule wins is the one saved and used unchanged at test time.

**Threshold selection method:** grid search over a fixed-threshold rule (`t` from
0.20 to 0.95, step 0.025) and the expected-F0.5 top-k rule (with a probability
floor of 0.0 to 0.5 below which a candidate is never kept, regardless of the
expected-F0.5 calculation), selecting whichever configuration scores best on the
50,000-entity tune set (see Validation protocol above) under the exact macro F0.5
formula. Chosen rule for this submission: a **plain probability
threshold at p ≥ 0.725**, applied after one-owner assignment. The curve was flat:
every threshold between 0.60 and 0.80 scored 0.968-0.9688, and the expected-F0.5
top-k rule scored 0.9678-0.9680 across its floor settings, consistently a little
below the plain threshold, so the simpler rule was kept rather than the more
elaborate one. Oracle ceiling on the tune set (macro F0.5 if every true candidate
that survived blocking were labelled perfectly): **0.9876**. The
gap from 1.0 to this ceiling is entirely blocking recall on this set; the further
gap from the ceiling down to the achieved 0.9688 is what the classifier and
decision rule leave on the table.

### Handling France with no training data

France appears only in the test set, with zero labelled French training pairs. Three
design choices make this workable without any France-specific code path:

1. **Country-agnostic features only.** No feature anywhere in `features.py` or
   `matcher.py` reads or encodes the `country` column; every feature is a
   similarity between two text fields. A model trained purely on US/India pairs
   therefore has no France-specific parameter to be missing.
2. **French patterns folded into normalisation, not modelling.** `normalize.py`
   already recognises French legal-form tokens (`sarl`, `eurl`, `sas`, `sasu`, `sa`,
   `sci`, `snc`, `ei`, `scop`, `selarl`) and French address abbreviations (`r` to
   `rue`, `allee`, `impasse`, `chemin`/`chem`, `faubourg`) alongside the US/India
   ones, so French records get the same quality of `core`/`legal`/`addr` fields as
   US or Indian records before any similarity score is computed.
3. **TF-IDF statistics are fit per country on the data being blocked, unsupervised.**
   `blocking.py` fits a fresh `TfidfTransformer` on whatever country's records are
   passed to `block_country()`, computed purely from term frequencies in that
   country's own test records. No document-frequency statistic is carried over from
   the US/India training data, and no label is used anywhere in this step, so
   France's blocking quality depends only on France's own test-set text, not on
   having seen France during training.

Because of this, the only training-data dependency for France is the LightGBM
model's learned feature-to-probability mapping (e.g. "high name similarity plus
address-number agreement usually means a match"), which is a hypothesis that
generic string-similarity patterns transfer across countries. `pipeline.py`'s
`write_submission()` now logs, on every `predict`/`select` run, the predicted
singleton rate and mean matches per Source 1 entity broken down by country
specifically so this hypothesis can be sanity-checked against the US/India numbers
without labels; see the code README's "Next iteration ideas" for turning that log
line into an automatic check instead of a manual read.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** **0.9688** on the 50,000-entity
  tune set (US + India only, using the tuned decision rule; see Validation protocol
  in Section 4 for why this needs the full 2.21M-entity scoring pass to be honest).
  By country: **US 0.9793**, **India 0.9529**. India scores lower, consistent with
  its heavier noise load (24% non-Latin-script names, transliterated legal forms
  and state names).
- **Ceiling on the tune set:** **0.9876** macro F0.5 if the matcher were perfect on
  the candidates blocking produced (the gap to 1.0 is blocking recall).
- **Precision/recall of the final matching pipeline (first full run):** precision
  **99.2%**, recall **93.1%** of all true pairs. Of the roughly 6.9 points of
  missed recall, about 3.3 points are lost at blocking (true pairs that never
  became a candidate at all, consistent with the 96.65% blocking recall in Section
  3), and the rest are pairs that did reach the classifier but scored below the
  0.725 threshold or lost the one-owner competition to a stronger claim.
- **Common false positives (wrong merges):** a meaningful share (about 0.8% of
  predicted pairs) are cases where the label appears arbitrary given the text:
  after normalisation the pair is essentially identical to a labelled true match
  elsewhere, yet the ground truth calls it a non-match. Example: for Source 1
  `Cascade American Partners | 247 Millville Avenue, Hamilton, OH`, the record
  `CASCADE AMERICAN PARTNERS CORP | 247 MILLVILLE AVE` is a labelled match, while
  `Cascade American Partners Co | 247 MILLVILLE AVE` at the exact same address is
  labelled a non-match. The synthetic-decoy generator appears to have produced some
  decoys that are textually indistinguishable from real noisy copies; this slice of
  error is irreducible for any text-similarity method, ours included (see the
  "label ambiguity" note in the code README's Next Iteration Ideas: not worth
  chasing further). Overall precision is 99.2%, so wrong merges are rare.
- **Common false negatives (missed matches):** three recurring patterns. (1) The
  business name replaced by what looks like an unrelated token at the same address,
  e.g. Source 1 name "Modern It" against a true Source 2/3 match written as
  "Pyralum": name similarity is near zero and the model scores it around p ~ 0.2,
  well under threshold, because "same address, unrelated name" also describes a
  different business at the same address, which the decoys make common. (2) An empty address on
  one side with an otherwise identical name: these land at p ~ 0.6-0.7, just under
  the 0.725 cutoff, because the address-similarity features are `NaN` (missing, not
  matched) and the name signal alone is not quite enough to clear the bar. (3)
  Partial addresses with the house number missing entirely, which weakens both the
  address fuzzy-match score and every `num_*` feature at once. All three point the
  same direction for future work: better use of address-only or name-only evidence
  when the other side is thin (see the code README's Next Iteration Ideas).

## 6. Conclusion

The pipeline treats entity resolution as blocking (get recall cheaply) followed by
classification and a metric-aware decision rule (get precision where it is scored).
Every noise pattern found in EDA (transliteration, decoys, digit corruption,
country-specific abbreviations) maps to a specific, testable piece of code
(`normalize.py`'s phonetic key and legal-form dictionary, `features.py`'s
number-truncation and legal-conflict features) rather than a black-box guess, and
the decision rule directly targets the competition's own F0.5 formula instead of a
generic accuracy proxy. The main lesson from the noise catalogue is that "same
business, sloppy data entry" and "different business, planted as a decoy" often
look identical on name similarity alone, and only resolve once address-number logic
and legal-form conflict are added as separate signals for the model to combine.

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/`, with all
source in `src/` (`normalize.py`, `blocking.py`, `features.py`, `matcher.py`,
`pipeline.py`), a `README.md` with exact reproduction steps, and a pinned
`requirements.txt`. Entry point:

```bash
python src/pipeline.py train      # normalise -> block -> features -> LightGBM -> tune decision rule
python src/pipeline.py predict    # same stages on test -> output/matching_results.tsv + candidate_pairs.tsv
```

Each stage caches its intermediate result under `--work` (normalised records,
per-country candidate pairs, feature tables, the saved model and decision rule), so
re-running after a small code change only needs to recompute the affected stage
onward via `--force`. `normalize.py`, `features.py`, and `matcher.py` each carry a
runnable `demo()` self-check (`python src/<module>.py`) asserting behaviour on the
exact noise patterns catalogued in Section 2.1. Full details, flag reference, and
the file-by-file map are in `code/business_entity_resolution/README.md`.

### B. Additional Results

Macro F0.5 on the 50,000-entity tune set as a function of the decision threshold
(after one-owner assignment). The curve is flat around the optimum, so the choice
is robust:

| threshold | 0.50 | 0.60 | 0.65 | 0.70 | **0.725** | 0.75 | 0.80 | 0.85 | 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro F0.5 | 0.9667 | 0.9681 | 0.9685 | 0.9687 | **0.9688** | 0.9686 | 0.9680 | 0.9665 | 0.9639 |

Expected-F0.5 top-k rule, for comparison: 0.9678 to 0.9680 depending on the floor.

Predicted profile on the test set (no labels, so this is a sanity check that the
model behaves the same in France, which it never saw in training). For reference,
5.6% of training Source 1 entities are true singletons, with about 3.5 true matches
on average, and our recall is about 93%:

| country | Source 1 entities | predicted singletons | mean predicted matches |
| --- | --- | --- | --- |
| France | 259,452 | 5.6% | 3.22 |
| India | 809,986 | 6.3% | 3.20 |
| US | 663,106 | 5.7% | 3.33 |

### C. Code Review Findings

Two independent reviewers audited `normalize.py`, `features.py`, and `matcher.py`/
`pipeline.py` against the noise catalogue in Section 2.1. Confirmed issues and the
fixes applied:

- **State names matched inside street names.** The old address normaliser matched
  state/region names anywhere in the text, so "Washington Street" became "wa st"
  (Washington treated as a state), affecting roughly 6% of records. Fixed by
  matching only whole comma-separated address components (`normalize.py`'s
  `_state()`), so "Washington Street" and "Rue du Nord" are left alone while a
  segment that is genuinely just "Ohio" or "Gironde" still gets canonicalised.
- **"S/O" and "C/O" mishandling.** "care of" / "son of" markers (`c/o`, `s/o`,
  `w/o`, `d/o`) were previously torn apart token by token, leaving stray single
  letters, and the same single-letter drop list accidentally deleted genuine
  single-letter street names ("O Street" lost its "O"). Fixed with a dedicated
  regex that removes the whole `c/o`-style phrase before tokenising, and by
  removing bare single letters from the address drop-list.
- **Website-as-name handling was too blunt.** A business name that is literally a
  domain (`metrocomponents.com`) needs to keep its stem as the name; a domain
  appended after the real name (`West Ltd | www.westltd.com`) is noise and should
  be dropped. Fixed by keying on whether an explicit `www.`/`http://` prefix is
  present: present means "appended link, drop it"; absent means "bare domain-like
  text, keep the stem".
- **Leetspeak repair over-corrected real digits.** The digit-to-letter table
  (`0`->`o`, `1`->`l`, etc.) previously touched digits at either edge of a token,
  which corrupted real numbers written into names ("24hr", "7eleven", "8th").
  Fixed by restricting the substitution to a digit with a letter on both sides
  (`br0thers` still fixes to "brothers"; "24hr" and "7eleven" are left untouched).
- **Missing text producing a fake similarity score.** A blank field (empty alias,
  empty first token) previously fell through to a manufactured value (e.g. counting
  two empty first-tokens as "equal"). `features.py` now returns `NaN` for every
  similarity feature, including `n_first_eq` and `n_alias`, whenever either side of
  the comparison is empty, consistent with the rest of the feature set.
- **Validation-protocol bias.** Described in Section 4's Validation protocol: an
  earlier version scored only the sampled tune-set entities before one-owner
  assignment, which could let a held-out entity keep a record an unsampled entity
  would actually have won on the full set. Fixed by scoring every train candidate
  pair (all 2.21M Source 1 entities) before restricting to the tune set. The
  measured effect was small (0.9689 before the fix, 0.9688 after), but the
  corrected number is the one trusted going forward.

One addition beyond bug-fixing: French address normalisation gained a
region/department table (`FR_REGIONS` in `normalize.py`), mapping a department to
its region so records naming either one still match: Gironde -> Nouvelle-Aquitaine
(`naq`), Nord and Pas-de-Calais -> Hauts-de-France (`hdf`), Loire-Atlantique ->
Pays de la Loire (`pdl`). This table was derived purely by reading the *unlabelled*
French test addresses for recurring region/department pairs, the same way the
US-state and Indian-state tables were built from patterns in the training data; no
external gazetteer or geocoding service was used, keeping it within the "no
external data" rule. `normalize.py` also gained a small name-synonym step
(`NAME_SYN`): "Etablissements"/"Etablissement" -> "ets", matching the existing "Ets"
abbreviation seen directly in the data.
