# Submission history

The rules require a version history of every leaderboard submission. Each file we
upload is produced by `python src/pipeline.py predict` (or `select --t X` for a
threshold variant), which appends a line to `output/manifest.tsv` with the file's
SHA-256 fingerprint, the git commit of the code and the decision rule. Copy that
line here when you upload, then add the leaderboard score.

Budget: 5 submissions per day for 3 days (15 total), 25-27 Sep 2026; the deadline is
27 Sep 11:59 PM. Per the organiser README, the final ranking uses the private
leaderboard only (the public board is a subset of the test set), so only coarse knobs
(the decision threshold) are probed on the public board; everything else is chosen on
our own validation split.

| # | date | uploaded by | file | sha256 (first 16) | code commit | rule | validation F0.5 | public LB F0.5 | notes |
|---|------|-------------|------|-------------------|-------------|------|-----------------|----------------|-------|
| 1 | 2026-09-25 14:28 IST | LycanAlan | matching_results.tsv | 486d3f430b1680f4 | 06b7ebd | threshold 0.725 | 0.9688 | 0.9570 | first full run |
| 2 | 2026-09-25 20:21 IST | LycanAlan | 1_v2_main/matching_results.tsv | 04fc9f0cdbab81fd | 0e3e243 | threshold 0.675, margin 0.4 | 0.9747 (India 0.9618, US 0.9832) | 0.963678 | v2: generator fingerprints, name ambiguity, runner-up margins (Ragh234) |
| 3 | 2026-09-25 20:31 IST | LycanAlan | 5_v2_diag_noFrance/matching_results.tsv | 2f1c4a7bcbe39543 | 0e3e243 | sub 2 with France rows emptied | - | 0.833986 | diagnostic: France = 0.0558 + (0.963678 - 0.833986) / 0.1498 = **0.922**; US+India on test = 0.971 vs 0.971 on validation |
| 4 | 2026-09-25 20:35 IST | LycanAlan | 2_v2_France0.85/matching_results.tsv | b8cbeb49da42e7db | 0e3e243 | threshold 0.675, margin 0.4; France 0.85 | - | 0.963831 | probe: France stricter, +0.000153 vs sub 2 = +0.0010 on France alone |
| 5 | 2026-09-25 22:50 IST | LycanAlan | v3/matching_results.tsv | 933e680deb5e121c | eb16c46 + 7a172ef (iter2 on main) | threshold 0.725, margin 0.4 | 0.9791 (India 0.9717, US 0.9841) | 0.96974 | v3 = v2 + number x name-sound blocking + normalise fixes; France = (0.96974 - 0.83088) / 0.1498 = 0.927 (v2: 0.922) |
| 6 | 2026-09-26 ~12:00 IST | LycanAlan | v4s2/matching_results.tsv | 6c630170bd480983 | 0a0579c (v4 + stage 2) | stage 2, threshold 0.675 | 0.9807 (stage-2 out-of-fold) | 0.96795 | v4 = v3 + house-number x address-word channel (France fix); France read-out ~0.906 if US/India on test = validation |
| 7 | 2026-09-26 ~12:05 IST | LycanAlan | v6s2/matching_results.tsv | 4649b37999b944e3 | 5f438a0 + 0a0579c | stage 2, threshold 0.675 | 0.9819 (India 0.9762, US 0.9857) | 0.96445 | v6 = v4 + twin features + phonetic fixes; France read-out ~0.874 |
| 8 | 2026-09-26 ~12:25 IST | LycanAlan | r4_night/matching_results.tsv | 10f7be5155dce48e | 2a930eb (night-raghav, Ragh234) | 3-model ensemble, bagged stage 2 | 0.9832 (India 0.9782, US 0.9865) | 0.965992 | v6 + character/twin retrieval, handle/number/address features; France read-out ~0.876 |
| 9 | 2026-09-26 ~16:10 IST | LycanAlan | h1_r4usin_v3fr/matching_results.tsv | 0a6f0462cbfbe8d1 | sub 8 US/India + sub 5 France | - | - | 0.964911 | diagnostic: France from v3 LOWERS the score, so France was not the problem; US/India on test ~0.009 below validation |
| 10 | 2026-09-26 ~22:15 IST | LycanAlan | ce1/matching_results.tsv | 78e44f17027a396e | v6 + xlm-roberta-base cross-encoder (Kaggle) | blend of logits (v6 p, cross-encoder), t 0.35, margin 0.2; unlabelled-country house-number rule | 0.9854 (India 0.9815, US 0.9879) | **0.977906** | best so far; France read-out ~0.941 if US/India on test = validation |

Upload tip (Windows): if the portal hangs at "Please Wait 0" / Bad Request, Windows has no
MIME type for `.tsv`. Fix once per user, then fully restart the browser:
`reg add "HKCU\Software\Classes\.tsv" /v "Content Type" /t REG_SZ /d "text/tab-separated-values" /f`

Reading sub 1: test is 46.8% India / 38.3% US / 15.0% France (S1). Re-weighting our
per-country validation (US 0.9793, India 0.9529) to that mix gives ~0.965 for US+India.
If test US/India behave like validation, France would be ~0.90 on the public split, but
that is an inference, not a measurement: the model's confidence profile on France test
pairs is the same as on US (52% of records matched at p>=0.99, same uncertain band).
A France-blanked diagnostic submission would measure it directly.

Reading sub 3 (France-blanked copy of sub 2): emptying a country's rows turns each of its S1
scores into 1 if the S1 has no true match, else 0. So LB(sub 2) - LB(sub 3) = w * (F_France - s),
with w = 0.1498 (France share of test S1) and s = 0.0558 (share of S1 with no match: 0.0558 in
both train countries, and v2 predicts 0.0568 empty for France). France scores **0.922**; US+India
together score 0.971 on test, the same as our validation predicts for that mix (0.971). Our
validation is trustworthy for US/India, and France is the whole leaderboard gap. To reach 0.98+
both need work: France 0.92 -> 0.97 is worth +0.0075, US+India 0.971 -> 0.985 is worth +0.012.

Reading subs 6-8: validation went up (0.9791 -> 0.9832) while the leaderboard went down
(0.96974 -> 0.96445..0.96795). Either France got worse, or US/India on test stopped tracking
validation. Test facts: 13.9% of French S1 share an exact address with another S1 (US 5.3%,
India 6.1%, train 6-7%), so address-based channels and twin/address features see a France
far outside training; test US is sparser than train US (name shared by another S1: 29% vs 36%).
Next: a hybrid file (US/India from sub 8, France from sub 5) separates the two.

Reading subs 9-10: test has ~2x the non-matching records per S1 (5.8 vs 4.7) and 2.4x the S1 with a
nearby-house-number candidate; dataset-count features, the 'margin over next-best S1' features and stage 2
over-state test performance. A labelled pseudo-test built from train (test-like density + synthetic
nearby-number decoys) reproduces the gap (validation 0.9819 -> 0.9708) and ranks stage 2 below stage 1.
A text model judging each pair on content (cross-encoder) transfers: sub 10 = +0.0082 over sub 5.
