# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** {{TEAM_NAME}}
**Team Members:** {{TEAM_MEMBERS}}
**Submission Date:** {{SUBMISSION_DATE}}

---

## 1. Executive Summary

We resolve business entities across three noisy sources with a blocking + classifier
pipeline: normalise text, retrieve a short candidate list per Source 1 entity with
hashed TF-IDF cosine similarity (blocking), score every candidate pair with a
LightGBM model trained on similarity features, then pick how many candidates to keep
per entity with a rule tuned directly against the macro F0.5 metric instead of a
fixed probability threshold. The pipeline is country-agnostic by construction (no
country feature is ever passed to the model), which is what lets it handle France in
the test set despite having zero French training examples. On a held-out validation
split of training S1 entities, this reaches a macro F0.5 of {{VAL_F05}}.

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
- The final decision rule does not threshold probabilities at a fixed cutoff. It
  computes the expected F0.5 of keeping the top k candidates for each Source 1
  entity, for every k including 0 (predict singleton), and keeps whichever k
  maximises that expected score, choosing between this rule and a plain threshold by
  whichever actually scores higher on held-out validation entities under the exact
  competition formula.
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
- Word unigram+bigram hashed TF-IDF ran about 3.5x faster than character n-grams at
  the same recall, which is why it is the final representation.
- **Final configuration:** reverse top-3 unioned with forward top-10 per source,
  query-side pruning at document frequency > 0.2%. Measured recall: 97.9% of true
  pairs survive on a US sample, at about 20 candidates per Source 1 entity. Recall
  on the full training run (all countries, this exact configuration): **{{BLOCK_RECALL}}**.

**Candidate pairs generated.** Total candidate pairs on the test set (the exact set
handed to the matching model, and what is written to `candidate_pairs.tsv`):
**{{N_CAND_PAIRS_TEST}}**.

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

Training uses 300,000 Source 1 entities' worth of candidate pairs (`--train-s1`,
default) with 100,000 held out for validation (`--valid-s1`). The top features by
gain on this run: **{{TOP_FEATURES}}**.

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
0.20 to 0.95) and the expected-F0.5 top-k rule (with a probability floor of 0.0 to
0.5 below which a candidate is never kept, regardless of the expected-F0.5
calculation), selecting whichever configuration scores best on 100,000 held-out
validation Source 1 entities under the exact macro F0.5 formula. Chosen rule for
this submission: **{{THRESHOLD_RULE}}**. Oracle ceiling (macro F0.5 if every true
candidate that survived blocking were labelled perfectly): **{{CEILING_F05}}**; the
gap between this ceiling and the achieved score attributes loss to the classifier
and decision rule rather than to blocking recall.

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
generic string-similarity patterns transfer across countries. See the "France
sanity checks" idea in the code README for how we plan to validate this once
predictions are available.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** {{VAL_F05}} (on 100,000 held-out training
  Source 1 entities, US + India only, using the tuned rule above)
- **Blocking recall ceiling on the same validation entities:** {{CEILING_F05}}
- **Common false positives (wrong merges):** expected to concentrate on the
  planted-decoy pattern from Section 2.1: a candidate with a very high name
  similarity and a house number within the `num_near` threshold (≤20) but wrong,
  particularly when the differing token is a legal suffix the *other* side lacks
  entirely (so `leg_conflict` cannot fire, since that feature only triggers when
  both sides carry a legal tag and disagree). This is the "Hong Management" vs.
  "Hong Management Inc" pattern.
- **Common false negatives (missed matches):** expected to concentrate on pairs
  that combine two or more heavy noise sources at once (e.g. a non-Latin-script
  Indian name with no address number at all), where blocking may rank the true
  pair outside `k_rev`/`k_fwd` because the shared, distinguishing vocabulary
  between the pair is thin. This is the roughly 2 percentage points of true pairs
  that blocking does not retrieve even at 97.9% measured recall.

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

Charts and tables to attach once a full pipeline run completes: the LightGBM
feature-importance plot behind {{TOP_FEATURES}}, the expected-F0.5-vs-threshold
curve used to pick {{THRESHOLD_RULE}}, and the France-vs-US-vs-India distribution
of predicted matches per Source 1 entity described as a sanity check in the code
README's "Next iteration ideas" section.

---

**Note:** This document follows the required structure of the official
`Documentation_template.md`. Placeholders in `{{DOUBLE_BRACES}}` mark values that
depend on an actual training/prediction run (not executed here per instructions)
and should be filled in from that run's logs before the final submission zip is
assembled.
