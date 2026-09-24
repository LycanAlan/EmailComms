# Submission history

The rules require a version history of every leaderboard submission. Each file we
upload is produced by `python src/pipeline.py predict` (or `select --t X` for a
threshold variant), which appends a line to `output/manifest.tsv` with the file's
SHA-256 fingerprint, the git commit of the code and the decision rule. Copy that
line here when you upload, then add the leaderboard score.

Budget: 5 submissions per day for 3 days (15 total). Final ranking uses both the
public and private leaderboards, so only coarse knobs (the decision threshold) are
probed on the public board; everything else is chosen on our own validation split.

| # | date | uploaded by | file | sha256 (first 16) | code commit | rule | validation F0.5 | public LB F0.5 | notes |
|---|------|-------------|------|-------------------|-------------|------|-----------------|----------------|-------|
