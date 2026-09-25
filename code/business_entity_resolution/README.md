# Business Entity Resolution

## Overview

This pipeline matches noisy business records across three sources for the Amazon ML
Challenge 2026 "Business Entity Resolution" task. Source 1 (S1) is a deduplicated
reference of ~1.7M businesses; Source 2 and Source 3 (S2/S3) together contribute
~10M noisy records (typos, transliteration, abbreviations, decoys) that may or may
not belong to any S1 entity. For every S1 entity we find all matching S2/S3 records.
The approach is blocking + classifier: normalise text, retrieve a short candidate
list per S1 with TF-IDF cosine similarity, score every candidate pair with a
LightGBM model trained on hand-built similarity features, then apply a decision
rule tuned directly against the competition's macro F0.5 metric to decide how many
candidates (if any) to keep per S1 entity. It never uses external data, APIs, or an
LLM, and the only trained model is LightGBM (MIT licensed, a tree ensemble, not a
parameter-counted neural net), satisfying the challenge's licensing and size rules.

## Setup

Requires Python 3.12.

```bash
pip install -r requirements.txt
```

Dependencies (all MIT/BSD/Apache licensed, see `requirements.txt`): anyascii,
lightgbm, numpy, pandas, pyarrow, RapidFuzz, scikit-learn, scipy, sparse-dot-topn.

## Reproducing the outputs

Run from `code/business_entity_resolution/`:

```bash
python src/pipeline.py train
python src/pipeline.py predict
```

`train` fits the LightGBM model and the decision rule on `dataset/train/` and saves
both to `--work/model/`. `predict` loads that saved model, runs it over
`dataset/test/`, and writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`.

To try a different probability cutoff without re-scoring 10M+ test pairs (e.g. to
probe the leaderboard with a slightly more or less conservative rule):

```bash
python src/pipeline.py select --t 0.65
```

This reads the test-set probabilities `predict` already saved to
`--work/test_scored.parquet`, applies a plain threshold at `p >= 0.65`, and writes
`output/matching_results_t0.65.tsv` in seconds. It never touches the model or the
candidate set, so it is only useful for exploring the threshold, not for a different
model or feature set. To move one country's cutoff alone (France has no training
labels, so its best cutoff can only be probed on the leaderboard):

```bash
python src/pipeline.py select --t 0.7 --t-country France=0.85
```

writes `output/matching_results_t0.7_m0_France0.85.tsv`.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `command` | (required) | one of `prepare`, `block`, `train`, `predict`, `select` |
| `--t` | `0.7` | `select` only: probability threshold for a new `matching_results_t<t>.tsv` variant, reusing the test scores saved by the last `predict` (no re-scoring) |
| `--margin` | `0` | `select` only: keep a pick only if the runner-up S1 for that record scored at least this much lower |
| `--t-country` | `[]` | `select` only: per-country overrides of `--t`, e.g. `France=0.85` |
| `--data` | `<repo_root>/6ab10eb3b23ba_student_resource/student_resource/dataset` | dataset root, expects `train/` and `test/` subfolders |
| `--work` | `~/er_work` | cache directory for normalised data, candidate pairs, features, and the saved model. Keep this off OneDrive/cloud-synced folders: parquet writes under a syncing folder are slow and can be picked up mid-write. |
| `--out` | `<repo_root>/output` | where `matching_results.tsv` and `candidate_pairs.tsv` are written |
| `--split` | `train` | which split to normalise/block; only used by the standalone `prepare`/`block` commands |
| `--force` | `[]` | any of `prepare`, `blocking`, `features`: recompute that stage instead of reusing its cache (see below) |
| `--jobs` | `7` | worker processes for normalisation and blocking |
| `--k-rev` | `3` | S1 candidates kept per S2/S3 record (reverse retrieval) |
| `--k-fwd` | `10` | S2/S3 candidates kept per S1 record, per source (forward retrieval) |
| `--prune` | `0.002` | query-side max document frequency; terms commoner than this fraction of rows are dropped before the sparse similarity search |
| `--train-s1` | `300000` | number of S1 entities used to fit the LightGBM model |
| `--valid-s1` | `100000` | held-out S1 entities: half early-stop LightGBM, the other half choose the decision rule and give the reported validation F0.5 |
| `--chunk` | `3000000` | pairs per chunk when scoring test candidates (bounds peak RAM) |

`<repo_root>` is resolved from `pipeline.py`'s own location (three levels up from
`src/`), so the defaults work as long as the folder layout from the submission zip
is kept intact.

### Caching and `--force`

Every stage writes its result under `--work` and is skipped on the next run if that
file already exists:

- `{split}_norm.parquet`: normalised S1+S2+S3 records (from `prepare`)
- `{split}_extras.parquet`: raw-text fingerprints + name ambiguity per record (from
  `prepare`, see `fingerprints.py`)
- `{split}_cands_<country>.parquet`: one file per country (from `block`). A cache
  written before the runner-up margin features existed gets them added on load.
- `train_feats_v<version>_<train_s1>_<valid_s1>.parquet`: features of the sampled
  train S1 entities (from `train`; the file name changes with the feature-set version
  and the sample sizes, so a stale table is never reused). Test features are computed
  in chunks and never stored.
- `tune_scored.parquet`: probabilities + labels of the rule-tuning S1 entities, for
  error analysis
- `model/lgb.txt` + `model/rule.json`: the trained model and chosen decision rule
  (from `train`; always overwritten on retrain)
- `test_scored.parquet`: probability for every test candidate pair (from `predict`).
  This is what lets `select` produce a new threshold variant in seconds instead of
  re-scoring 10M+ pairs.

Every file `predict`/`select` writes to `--out` is also logged, one row per write,
to `output/manifest.tsv` (created on first use): timestamp, file name, the sha256 of
that exact file, the short git commit hash of the code that produced it (or
`no-git` if the code folder isn't a git checkout or git isn't installed), the decision rule used, and the
match count. This is the automatic, code-side half of the submission version
history the challenge rules require; see `SUBMISSIONS.md` at the repository root
for the human-maintained half (which of these files were actually uploaded, when,
and what public/private score each got). The leaderboard budget is 5 uploads/day
for 3 days, and ranking uses both the public and private board, so most of that
budget is spent probing the decision threshold with `select` (see above) rather
than retraining, and each upload gets one row in `SUBMISSIONS.md`. Commit the
code before running `predict`/`select`, so the manifest points at a real commit.

Caching only checks whether *its own* output file exists, it does not check whether
an earlier stage changed. If you edit `normalize.py` and want that reflected, you
must force every stage from that point on yourself, e.g.:

```bash
python src/pipeline.py train --force prepare blocking features
```

Forcing only `blocking` also recomputes features (they depend on the candidate set),
but forcing `prepare` alone will NOT automatically recompute `blocking` or
`features`: pass all the stages you need explicitly.

### Expected runtime

On a 16GB RAM / 12-thread laptop:

- `python src/pipeline.py train`: **about 58 minutes end to end** (normalise ~2 min,
  blocking ~25 min, features ~2 min, LightGBM fit ~5 min at 1011 rounds, scoring all
  56M train candidate pairs for the honest validation pass ~25 min)
- `python src/pipeline.py predict`: **about 41 minutes (normalise 2 min, blocking 15 min, scoring 48M pairs 23 min, writing 1 min)**

(about 41 minutes (normalise 2 min, blocking 15 min, scoring 48M pairs 23 min, writing 1 min) is a placeholder, fill in after a timed run on the target machine)

## Self-checks

`normalize.py`, `features.py`, and `matcher.py` each carry a small `demo()` with
hard asserts on the exact noise patterns seen in the data (leetspeak, transliterated
legal suffixes, dropped/padded digits, decoy house numbers, "formerly known as",
websites-as-names, etc.). Run them directly to sanity-check the code without
touching the full dataset:

```bash
python src/normalize.py
python src/features.py
python src/matcher.py
```

Each prints `<module>.demo OK` and exits 0 on success, or raises an `AssertionError`
pointing at the failing case.

## Validating the submission

The organiser-provided validator is stdlib-only and lives in the problem statement
folder. From the repository root:

```bash
cd 6ab10eb3b23ba_student_resource/student_resource
python utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Prints `PASS` (exit 0) when the files are safe to submit, or a numbered list of
issues to fix (exit 1). Add `--check-ids` for the (memory-heavier) check that every
matched/candidate ID actually exists in the test Source-2/3 files.

## File-by-file map of `src/`

| File | Role |
| --- | --- |
| `normalize.py` | Turns a raw `business_name`/`business_address` into canonical, comparable fields: `core` (name minus legal form/filler/noise), `alt` (the "Y" in "formerly known as Y" / "dba Y"), `legal` (legal-form family, e.g. `ltd`, `llp`, `sarl`), `phon` (consonant-skeleton phonetic key, robust to typos and transliteration), `nosp` (`core` with spaces stripped, matches names against website-style handles), `addr` (address with abbreviations/state names canonicalised), `nums` (numbers pulled out of the address, leading zeros stripped), `nl` (1 if the raw name used a non-Latin script). Pure functions, no I/O. |
| `blocking.py` | Candidate generation. Builds hashed TF-IDF (word uni+bigram) vectors per country over name and address text and retrieves, per S1 record, its most similar S2/S3 records (forward) and, per S2/S3 record, its most similar S1 records (reverse), unions the two, then attaches TF-IDF cosine, rank/gap "context" features and runner-up margins (how far a pair is ahead of the best competing claimant; `rgap` alone is 0 for a winner whether it wins clearly or ties) used later by the classifier. `block_country()` is the entry point, called once per country. |
| `features.py` | Pairwise similarity features for a candidate pair: string-distance and token-overlap scores on the name (`n_*`), legal-form agreement/conflict (`leg_*`), address similarity (`a_*`), and address-number logic that separates true noise (dropped digits, zero-padding) from decoys (nearby-but-different house numbers) (`num_*`). `pair_features()` is the entry point; scoring runs in RapidFuzz's C++ layer across all cores. |
| `fingerprints.py` | Record-level signals that normalisation erases. Raw-text "generator fingerprints": the synthetic noise (empty address, address without numbers, website or alias in the name, lowercase, leetspeak...) is applied to true copies far more often than to decoys, while decoys are longer and keep their legal form. Name ambiguity: how many S1 entities share a record's cleaned name or phonetic key, and how much of a name is made of words used by no S1 name (made-up replacement names). All computed per country from the split's own records, without labels. |
| `matcher.py` | The LightGBM classifier and the decision rule. `fit()` trains the binary "same business" model; `predict()` scores pairs; `one_owner()` keeps each S2/S3 record for only the S1 entity that scored it highest (run once over ALL S1 entities); `select()` then turns probabilities into final per-S1 lists, either by a plain threshold or by the expected-F0.5 top-k rule; `tune_decision()` picks whichever rule scores best, using the exact competition metric, on held-out S1 entities; `macro_f05()` computes that metric; `save()`/`load()` persist the model and chosen rule. |
| `pipeline.py` | CLI entry point. Reads the raw TSVs, orchestrates `prepare`/`block`/`train`/`predict`/`select` with on-disk caching under `--work`, and writes the required output files. `score_pairs()` runs the model over a candidate set in RAM-bounded chunks (used by both `train`'s honest validation pass and `predict`); `write_submission()` applies `one_owner()`+`select()`, writes a matches file, logs per-country singleton rate / mean matches (the France sanity check), and appends a row to `output/manifest.tsv` (sha256, git commit, rule, match count) for every file it writes. Also the one place with a Windows-specific gotcha (see below). |
| `requirements.txt` | Pinned dependency versions. |

## How it works, in plain words

The same business can be written a dozen different ways across three independent,
messy sources. The pipeline turns "find every S2/S3 record that is secretly the same
business as this S1 record" into five smaller, boring steps.

**1. Normalise.** Before comparing anything, strip away the noise that makes two
identical businesses look different: legal suffixes ("Pvt Ltd" vs "Private
Limited"), scripts (Hindi/Kannada names get transliterated), typos and leetspeak
("Br0thers"), phone numbers and "(ID: 23374)" tags stuck onto a name, "formerly
known as" aliases, and address abbreviations ("Rd" vs "Road", "OH" vs "Ohio").
*Why:* every later comparison is a text-similarity score, and text similarity on
unnormalised strings is comparing formatting, not meaning. Do this once, cheaply,
instead of re-deriving it inside every feature.

**2. Block (candidate generation).** You cannot compare 1.7M S1 records against 10M
S2/S3 records pairwise: that's ~17 trillion comparisons. Instead, represent every
record as a TF-IDF vector (hashed word unigrams+bigrams over name and address,
computed separately per country) and use fast sparse cosine similarity to fetch a
short list of maybe 20 plausible candidates per S1 record. Two directions are
combined: each S2/S3 record also looks up its own best-matching S1 records (since
each S2/S3 record belongs to at most one S1 entity, this direction is very
precise), and each S1 record looks up its best-matching S2/S3 records per source
(this catches the S1's true owner even when a generic-sounding S2/S3 record ranked
it low from the other direction). *Why:* blocking sets a hard ceiling on recall.
A true match that never becomes a candidate can never be found downstream, no
matter how good the classifier is, so it is worth spending effort making the
candidate list both cheap and high-recall.

**3. Features.** For every (S1, candidate) pair that survived blocking, compute
several dozen numeric similarity scores: how alike are the cleaned names (edit
distance, token overlap, phonetic skeleton), do the legal-form tags agree or
conflict, how alike are the addresses, and do the address numbers look like typical
noise (a dropped digit, a zero-padded ID) or like a decoy (same name, house number
off by 3). *Why:* a single similarity score can't separate "same business, sloppy
data entry" from "different business planted right next to it as a decoy". You
need several signals and let a model combine them, not a hand-tuned formula.

**4. LightGBM.** A gradient-boosted tree model turns each pair's feature vector into
a probability that the two records describe the same business. *Why LightGBM:* it
handles missing values natively (many features are `NaN` when, say, one side has no
address number), it trains fast on tens of millions of rows on a laptop, its feature
importances are easy to inspect, and, importantly for this challenge's rules, it
is MIT-licensed and is a tree ensemble, not a large neural network, so the "≤8B
parameters" model-size constraint is trivially satisfied.

**5. F0.5 decision.** Probabilities alone aren't an answer. You have to decide,
per S1 entity, which candidates (if any) to actually keep. Two rules apply first:
each S2/S3 record is only kept for the single S1 entity that scored it highest
(matches the data's real one-owner-per-record structure), and then, because F0.5
rewards precision twice as much as recall, the pipeline does not simply threshold
at 0.5. It computes, for every possible cutoff k (0, 1, 2, ...) of an S1 entity's
sorted candidate probabilities, the *expected* F0.5 you'd get by keeping the top k
(using the probabilities themselves as an estimate of how many are actually true
matches), and keeps whichever k scores highest, including k = 0, which is
"predict singleton". A plain probability threshold is tried too, and whichever rule
(threshold or expected-F0.5 top-k) scores better on held-out validation S1 entities,
measured with the exact competition formula, is the one saved and used.

How validation is kept honest: the sampled train S1 entities are split three ways
(300k fit the model, 50k early-stop it, 50k choose the rule and report the score).
Before scoring those last 50k, the model scores *every* train candidate pair (all
2.2M S1 entities), so the one-owner step sees exactly the competition it will see on
test. Scoring only the 50k would let each of them keep records that really belong to
an S1 entity outside the sample, which inflates recall and picks the wrong threshold.
*Why:* this
directly optimises the metric being graded, rather than a proxy like accuracy or a
fixed 0.5 cutoff, and it naturally handles the fact that different S1 entities have
very different numbers of true matches (0 to double digits).

## Windows gotcha

`lightgbm` must be imported before `scikit-learn`. On Windows, importing them in
the opposite order crashes with an OpenMP runtime clash. `pipeline.py` imports
`lightgbm` first (with a `# noqa: F401` comment explaining why) for exactly this
reason: keep that import order if you touch the top of the file.

## Next iteration ideas

Ideas that did not make it into this submission but would likely help, roughly in
order of expected payoff for effort:

- **Learned transliteration dictionary.** Mine the training pairs for
  non-Latin-to-Latin name spellings that keep recurring (the way `LEGAL` in
  `normalize.py` already hardcodes a handful of transliterated legal-form tokens),
  instead of relying only on `anyascii` + the phonetic skeleton.
- **Cross-encoder re-ranking on uncertain pairs.** For pairs the LightGBM model
  scores in the uncertain band (p in roughly 0.3-0.9), re-score with a small
  Apache/MIT-licensed cross-encoder that reads the raw name/address text pair
  directly, and blend that into the final decision. Leave the confident pairs
  (p outside that band) to the cheap LightGBM score alone to keep runtime down.
- **Cluster-consistency features.** For an S1 candidate from S2, check whether it
  agrees (same address numbers, same legal form) with the S3 candidates already
  linked to the same S1. A true match usually agrees with the *other* true matches
  of its S1 entity, while a decoy usually doesn't.
- **Second-stage model on first-stage probabilities.** Add features like "gap to
  the best probability among this S1's other candidates" and "gap to the best
  probability this candidate got from any other S1" (partly present already as
  blocking-stage `*_gap`/`*_rgap` columns) and feed them, plus the first-stage `p`,
  into a small second model or a refit of the decision rule.
- **Blocking recall.** Try `--prune 0.005` (looser query-side pruning) and
  `--k-rev 5` (more reverse candidates) and re-measure recall vs. candidate-count
  cost. The current config gets 97.9% recall on a US sample; a few points more
  raises the ceiling every downstream stage is capped by.
- **France sanity checks (now logged automatically, still needs a human look).**
  `write_submission()` already prints, per country, the predicted singleton rate and
  mean matches per S1 every time `predict`/`select` runs, specifically so France
  (no training labels) can be eyeballed against US and India without extra work. It
  is still only a printed log line, not a gate: nothing stops a bad run from
  shipping. Next step is turning it into an actual check, e.g. fail loudly if
  France's singleton rate is more than a few points off the US/India range, rather
  than relying on someone reading the log.
- **Do not chase the label-ambiguity finding.** Some fraction of "false positives"
  (see the code review notes in the methodology doc) are pairs the normalised text
  makes genuinely indistinguishable from a real match, and the ground truth still
  calls them a non-match. No feature can fix a label that is arbitrary given the
  available text; spending time trying to claw back this last slice of precision is
  very likely wasted effort compared to the other ideas here.
- **Raise recall on empty-address and replaced-name matches.** The two recurring
  false-negative patterns are a true match with no address at all (name similarity
  alone lands just under the threshold) and a true match where the name was swapped
  for a seemingly unrelated token at the same address. Both need a signal beyond
  what `features.py` currently computes, e.g. weighting address-only agreement more
  when the name side is uninformative, or a feature that explicitly rewards "same
  address, different name" instead of only penalising it as a name mismatch.
- **More rounds / bigger training sample.** `--train-s1` defaults to 300k of the
  ~1.7-2.2M available S1 entities and LightGBM trains for up to 2000 rounds with
  early stopping; if training time allows, a larger sample and more rounds are a
  cheap way to squeeze out marginal gains.
