"""End-to-end entry point.

  python src/pipeline.py train      # normalise -> block -> features -> LightGBM -> tune decision rule
  python src/pipeline.py predict    # same on test -> output/matching_results.tsv + candidate_pairs.tsv

Each stage caches its result in --work, so after changing one stage only that
stage and the ones after it need recomputing:  --force features  (or blocking, prepare).
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm  # noqa: F401  MUST load before scikit-learn: on Windows the reverse order crashes LightGBM (OpenMP clash)
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

import blocking
import features
import matcher
import normalize

ROOT = Path(__file__).resolve().parents[3]
STR = pd.StringDtype("pyarrow")                     # compact strings: ~10x less RAM than Python objects
NORM_COLS = ["core", "alt", "legal", "phon", "nosp", "nl", "addr", "nums"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_tsv(path):
    t = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                       convert_options=pacsv.ConvertOptions(strings_can_be_null=False,
                                                            column_types={c: pa.string() for c in
                                                                          ["entity_id", "business_name", "business_address", "country"]}))
    return t.to_pandas(types_mapper={pa.string(): STR}.get)


def _norm_chunk(rows):
    out = pd.DataFrame([normalize.norm_name(n) + normalize.norm_addr(a) for n, a in rows], columns=NORM_COLS)
    out["nl"] = out.nl.astype(np.int8)
    return out.astype({c: STR for c in NORM_COLS if c != "nl"})


def load_norm(a, split):
    """All records of a split (S1, S2, S3 stacked), normalised. Row position = record key."""
    path = a.work / f"{split}_norm.parquet"
    if path.exists() and "prepare" not in a.force:
        return pd.read_parquet(path).astype({"id": STR, "country": STR, **{c: STR for c in NORM_COLS if c != "nl"}})
    log(f"prepare {split}: normalising")
    frames = []
    with Pool(a.jobs) as pool:
        for s in (1, 2, 3):
            raw = read_tsv(a.data / split / f"{split}_source{s}.tsv")
            rows = list(zip(raw.business_name.tolist(), raw.business_address.tolist()))
            chunks = (rows[i:i + 20_000] for i in range(0, len(rows), 20_000))
            n = pd.concat(pool.imap(_norm_chunk, chunks), ignore_index=True)
            n.insert(0, "id", raw.entity_id.to_numpy())
            n.insert(1, "src", np.int8(s))
            n.insert(2, "country", raw.country.to_numpy())
            frames.append(n)
            log(f"  source{s}: {len(n):,} records")
            del raw, rows
    norm = pd.concat(frames, ignore_index=True)
    norm.to_parquet(path)
    return norm


def load_candidates(a, split, norm):
    """Candidate pairs for every country, cached per country. q/i are row positions in norm."""
    out = {}
    for cty in norm.loc[norm.src == 1, "country"].unique():
        path = a.work / f"{split}_cands_{re.sub(r'[^A-Za-z0-9]+', '_', cty)}.parquet"
        if not path.exists() or "blocking" in a.force:
            log(f"blocking {split} / {cty}")
            d = norm[norm.country == cty]
            with Pool(a.jobs) as pool:
                c = blocking.block_country(d.reset_index(drop=True), a.k_rev, a.k_fwd, a.prune, pool)
            if c is None:
                continue
            pos = d.index.to_numpy()
            c["q"], c["i"] = pos[c.q.to_numpy()], pos[c.i.to_numpy()]
            c.to_parquet(path)
        out[cty] = path
    return out


def truth_pairs(a, norm):
    """Ground truth as a set of (q, i) row-position pairs + per-S1 true-match counts."""
    gt = read_tsv(a.data / "train" / "train_ground_truth.tsv")
    pos = pd.Series(np.arange(len(norm)), index=norm.id.to_numpy())
    s1 = np.repeat(gt.source1_entity_id.to_numpy(), gt.matched_entity_ids.str.count(",").to_numpy() + 1)
    m = np.concatenate(gt.matched_entity_ids.str.split(",").to_numpy())
    keep = m != ""
    q, i = pos[s1[keep]].to_numpy(), pos[m[keep]].to_numpy()
    n_true = pd.Series(0, index=pos[gt.source1_entity_id.to_numpy()].to_numpy())
    n_true = n_true.add(pd.Series(q).value_counts(), fill_value=0).astype(int)
    return q, i, n_true


def build_features(a, norm, cand_paths, s1_keep):
    """Feature table for the candidate pairs of the sampled train S1 rows."""
    path = a.work / f"train_feats_{a.train_s1}_{a.valid_s1}.parquet"      # the sample is fixed by seed + sizes
    if path.exists() and not {"features", "blocking"} & set(a.force):
        return pd.read_parquet(path)
    parts = []
    for cty, p in cand_paths.items():
        c = pd.read_parquet(p)
        c = c[np.isin(c.q.to_numpy(), s1_keep)]
        log(f"features train / {cty}: {len(c):,} pairs")
        parts.append(features.pair_features(c.reset_index(drop=True), norm))
    f = pd.concat(parts, ignore_index=True)
    f.to_parquet(path)
    return f


def score_pairs(a, norm, cand_paths, model):
    """P(match) for EVERY candidate pair, in chunks to bound RAM -> DataFrame(q, i, p)."""
    scored = []
    for cty, p in cand_paths.items():
        c = pd.read_parquet(p)
        log(f"scoring {cty}: {len(c):,} pairs")
        for s in range(0, len(c), a.chunk):
            f = features.pair_features(c.iloc[s:s + a.chunk].reset_index(drop=True), norm)
            scored.append(pd.DataFrame({"q": f.q.to_numpy(np.int32), "i": f.i.to_numpy(np.int32),
                                        "p": matcher.predict(model, f)}))
    return pd.concat(scored, ignore_index=True)


def cmd_train(a):
    norm = load_norm(a, "train")
    cands = load_candidates(a, "train", norm)
    tq, ti, n_true = truth_pairs(a, norm)
    s1_rows = np.flatnonzero(norm.src.to_numpy() == 1)
    key = lambda q, i: q.astype(np.int64) * len(norm) + i
    truth = key(tq, ti)

    # blocking recall = ceiling on what the matcher can ever find
    allc = pd.concat([pd.read_parquet(p, columns=["q", "i"]) for p in cands.values()])
    found = np.isin(truth, key(allc.q.to_numpy(), allc.i.to_numpy()))
    log(f"blocking: {len(allc):,} pairs ({len(allc) / len(s1_rows):.1f} per S1), recall {found.mean():.4f}")
    del allc

    # three disjoint groups of S1 entities: fit the model / early-stop it / choose + report the decision rule
    rng = np.random.default_rng(0)
    sample = rng.choice(s1_rows, min(a.train_s1 + a.valid_s1, len(s1_rows)), replace=False)
    fit_q, stop_q, tune_q = np.split(sample, [a.train_s1, a.train_s1 + a.valid_s1 // 2])
    f = build_features(a, norm, cands, np.concatenate([fit_q, stop_q]))
    f["y"] = np.isin(key(f.q.to_numpy(), f.i.to_numpy()), truth).astype(np.int8)
    is_stop = np.isin(f.q.to_numpy(), stop_q)
    model = matcher.fit(f[~is_stop], f[is_stop])
    del f

    # Honest evaluation: score EVERY train pair (all 2.2M S1), so the one-owner step sees the same
    # competition between S1 entities as it will on test, then judge only the untouched tune_q entities.
    owned = matcher.one_owner(score_pairs(a, norm, cands, model))
    ev = owned[np.isin(owned.q.to_numpy(), tune_q)].copy()
    ev["y"] = np.isin(key(ev.q.to_numpy(), ev.i.to_numpy()), truth).astype(np.int8)
    reachable = truth[np.isin(tq, tune_q) & found]                  # true pairs blocking kept
    ceiling = matcher.macro_f05(pd.DataFrame({"q": tq[np.isin(tq, tune_q) & found], "y": 1}), tune_q, n_true)
    log(f"tune set: {len(tune_q):,} S1, {len(reachable):,} reachable true pairs, ceiling F0.5 {ceiling:.4f}")
    rule = matcher.tune_decision(ev, tune_q, n_true)
    picked = matcher.select(ev, rule)
    cty = norm.country.to_numpy()
    for c in np.unique(cty[tune_q]):
        qs = tune_q[cty[tune_q] == c]
        log(f"  {c}: macro F0.5 {matcher.macro_f05(picked[np.isin(picked.q.to_numpy(), qs)], qs, n_true):.4f}")
    ev.to_parquet(a.work / "tune_scored.parquet")                     # for error analysis
    matcher.save(model, rule, a.work / "model")
    log(f"saved model + decision rule {rule} to {a.work / 'model'}")


def cmd_predict(a):
    norm = load_norm(a, "test")
    cands = load_candidates(a, "test", norm)
    model, rule = matcher.load(a.work / "model")
    s1_rows = np.flatnonzero(norm.src.to_numpy() == 1)
    scored = score_pairs(a, norm, cands, model)
    scored.to_parquet(a.work / "test_scored.parquet")               # lets `select` make variants in seconds
    a.out.mkdir(parents=True, exist_ok=True)
    write_lists(a.out / "candidate_pairs.tsv", "candidate_entity_ids", norm.id.to_numpy(), s1_rows,
                scored.q.to_numpy(), scored.i.to_numpy())
    write_submission(a, norm, scored, rule, "matching_results.tsv")


def cmd_select(a):
    """Another decision threshold on the saved test scores, e.g. to probe the leaderboard."""
    norm = load_norm(a, "test")
    scored = pd.read_parquet(a.work / "test_scored.parquet")
    write_submission(a, norm, scored, {"kind": "threshold", "t": a.t}, f"matching_results_t{a.t:g}.tsv")


def write_submission(a, norm, scored, rule, name):
    s1_rows = np.flatnonzero(norm.src.to_numpy() == 1)
    picked = matcher.select(matcher.one_owner(scored), rule)
    path = a.out / name
    write_lists(path, "matched_entity_ids", norm.id.to_numpy(), s1_rows, picked.q.to_numpy(), picked.i.to_numpy())
    log(f"wrote {path}: {len(picked):,} matches for {len(s1_rows):,} S1 entities, rule {rule}")
    # sanity check for France (no labels): its profile should look like the countries we trained on
    k = pd.Series(0, index=s1_rows).add(picked.q.value_counts(), fill_value=0)
    for c, g in k.groupby(norm.country.to_numpy()[s1_rows]):
        log(f"  {c}: {len(g):,} S1, predicted singletons {(g == 0).mean():.1%}, mean matches {g.mean():.2f}")
    # version history: which code + rule produced exactly which file
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip() or "no-git"
    except OSError:                                                  # git not installed (e.g. a grader's machine)
        commit = "no-git"
    manifest = a.out / "manifest.tsv"
    new = not manifest.exists()
    with manifest.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("created\tfile\tsha256\tcode_commit\trule\tn_matches\n")
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\t{sha}\t{commit}\t{json.dumps(rule)}\t{len(picked)}\n")
    log(f"  sha256 {sha[:16]}... logged in {manifest}")


def write_lists(path, col, ids, s1_rows, q, i):
    """One row per S1 entity (empty list allowed), ids comma-joined, no duplicates."""
    lists = pd.Series(ids[i]).groupby(q).agg(lambda x: ",".join(dict.fromkeys(x)))
    out = pd.DataFrame({"source1_entity_id": ids[s1_rows], col: lists.reindex(s1_rows).fillna("").to_numpy()})
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "block", "train", "predict", "select"])
    ap.add_argument("--t", type=float, default=0.75, help="select: probability threshold for the variant file")
    ap.add_argument("--data", type=Path, default=ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset")
    ap.add_argument("--work", type=Path, default=Path.home() / "er_work", help="cache dir (keep it off OneDrive)")
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    ap.add_argument("--split", default="train", help="for prepare/block only")
    ap.add_argument("--force", nargs="*", default=[], choices=["prepare", "blocking", "features"])
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--k-rev", type=int, default=3, help="S1 candidates kept per S2/S3 record")
    ap.add_argument("--k-fwd", type=int, default=10, help="S2 (and S3) candidates kept per S1 record")
    ap.add_argument("--prune", type=float, default=0.002, help="query-side max document frequency")
    ap.add_argument("--train-s1", type=int, default=300_000, help="S1 entities used to fit the model")
    ap.add_argument("--valid-s1", type=int, default=100_000, help="held-out S1 entities for scoring/tuning")
    ap.add_argument("--chunk", type=int, default=3_000_000)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.command == "prepare":
        load_norm(a, a.split)
    elif a.command == "block":
        load_candidates(a, a.split, load_norm(a, a.split))
    else:
        {"train": cmd_train, "predict": cmd_predict, "select": cmd_select}[a.command](a)


if __name__ == "__main__":
    main()
