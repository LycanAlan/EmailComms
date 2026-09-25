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

Upload tip (Windows): if the portal hangs at "Please Wait 0" / Bad Request, Windows has no
MIME type for `.tsv`. Fix once per user, then fully restart the browser:
`reg add "HKCU\Software\Classes\.tsv" /v "Content Type" /t REG_SZ /d "text/tab-separated-values" /f`

Reading sub 1: test is 46.8% India / 38.3% US / 15.0% France (S1). Re-weighting our
per-country validation (US 0.9793, India 0.9529) to that mix gives ~0.965 for US+India.
If test US/India behave like validation, France would be ~0.90 on the public split, but
that is an inference, not a measurement: the model's confidence profile on France test
pairs is the same as on US (52% of records matched at p>=0.99, same uncertain band).
A France-blanked diagnostic submission would measure it directly.
