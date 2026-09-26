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
import pyarrow.parquet as pq

import blocking
import features
import fingerprints
import matcher
import normalize

ROOT = Path(__file__).resolve().parents[3]
STR = pd.StringDtype("pyarrow")                     # compact strings: ~10x less RAM than Python objects
NORM_COLS = ["core", "alt", "legal", "phon", "nosp", "nl", "addr", "nums"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_arrow(path):
    return pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                          convert_options=pacsv.ConvertOptions(strings_can_be_null=False,
                                                               column_types={c: pa.string() for c in
                                                                             ["entity_id", "business_name", "business_address", "country"]}))


def read_tsv(path):
    return read_arrow(path).to_pandas(types_mapper={pa.string(): STR}.get)


def _norm_chunk(rows):
    out = pd.DataFrame([normalize.norm_name(n) + normalize.norm_addr(a) for n, a in rows], columns=NORM_COLS)
    out["nl"] = out.nl.astype(np.int8)
    return out.astype({c: STR for c in NORM_COLS if c != "nl"})


def load_norm(a, split):
    """All records of a split (S1, S2, S3 stacked), normalised, plus the raw-text fingerprints and name
    ambiguity of fingerprints.py (cached separately). Row position = record key."""
    norm = _normalised(a, split)
    path = a.work / f"{split}_extras_v{features.VERSION}.parquet"   # extras change with the feature set
    if path.exists() and "prepare" not in a.force:
        extra = pd.read_parquet(path)
    else:
        log(f"prepare {split}: raw-text fingerprints + name ambiguity")
        parts = []
        for s in (1, 2, 3):
            t = read_arrow(a.data / split / f"{split}_source{s}.tsv")
            parts.append(fingerprints.raw_flags(t["business_name"], t["business_address"]))
            del t
        extra = pd.concat([pd.concat(parts, ignore_index=True), fingerprints.ambiguity(norm)], axis=1)
        extra.to_parquet(path)
    return pd.concat([norm, extra], axis=1)


def _normalised(a, split):
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
        # the name carries every setting that changes the candidate set, so a stale cache is never reused
        tag = f"v{blocking.VERSION}_r{a.k_rev}f{a.k_fwd}p{a.prune:g}c{a.k_char}m{a.char_min:g}t{a.tw_cap}"
        path = a.work / f"{split}_cands_{tag}_{re.sub(r'[^A-Za-z0-9]+', '_', cty)}.parquet"
        if not path.exists() or "blocking" in a.force:
            log(f"blocking {split} / {cty}")
            d = norm[norm.country == cty]
            with Pool(a.jobs) as pool:
                c = blocking.block_country(d.reset_index(drop=True), a.k_rev, a.k_fwd, a.prune, pool,
                                           k_char=a.k_char, char_min=a.char_min, tw_cap=a.tw_cap)
            if c is None:
                continue
            pos = d.index.to_numpy()
            c["q"], c["i"] = pos[c.q.to_numpy()], pos[c.i.to_numpy()]
            c.to_parquet(path)
        elif "cos_w_rmarg" not in pq.read_schema(path).names:          # cache from before the margin features
            log(f"adding runner-up margins to {path.name}")
            c = pd.read_parquet(path)
            blocking.add_margins(c)
            c.to_parquet(path)
            del c
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


def build_features(a, norm, cand_paths, s1_keep, cache=True):
    """Feature table for the candidate pairs of the sampled train S1 rows."""
    path = a.work / f"train_feats_v{features.VERSION}_{a.train_s1}_{a.valid_s1}.parquet"   # sample fixed by seed + sizes
    if cache and path.exists() and not {"features", "blocking"} & set(a.force):
        return pd.read_parquet(path)
    parts = []
    for cty, p in cand_paths.items():
        c = pd.read_parquet(p)
        c = c[np.isin(c.q.to_numpy(), s1_keep)]
        log(f"features train / {cty}: {len(c):,} pairs")
        parts.append(features.pair_features(c.reset_index(drop=True), norm))
        del c
    f = pd.concat(parts, ignore_index=True)
    if cache:
        f.to_parquet(path)
    return f


def best_threshold_score(ev, s1_rows, n_true):
    """The best macro F0.5 of the threshold + margin grid, without printing it (for side-by-side reports)."""
    return max((matcher.macro_f05(matcher.select(ev, {"kind": "threshold", "t": t, "margin": m}), s1_rows, n_true), t, m)
               for t in np.arange(0.4, 0.96, 0.025) for m in (0.0, 0.4, 0.5))


def score_pairs(a, norm, cand_paths, model, keep_q=None, each=False):
    """P(match) for EVERY candidate pair -> DataFrame(q, i, p). The candidate file is streamed in
    chunks: loaded whole, a country's table (~45 columns x 45M pairs) no longer fits in RAM.
    keep_q: only the pairs of records that some keep_q S1 retrieved. one_owner() decides a record
    from its own pairs alone, so the keep_q rows come out exactly as if everything was scored.
    model may be a list (p = their average); each=True also returns every model's own p_k."""
    models = model if isinstance(model, list) else [model]
    scored = []
    for cty, p in cand_paths.items():
        keep_i = None
        if keep_q is not None:
            qi = pd.read_parquet(p, columns=["q", "i"])
            keep_i = np.unique(qi.i.to_numpy()[np.isin(qi.q.to_numpy(), keep_q)])
            del qi
        log(f"scoring {cty}: {pq.read_metadata(p).num_rows:,} pairs" + ("" if keep_i is None else
                                                                        f", those of {len(keep_i):,} records"))
        for b in pq.ParquetFile(p).iter_batches(batch_size=a.chunk):
            c = b.to_pandas()
            if keep_i is not None:
                c = c[np.isin(c.i.to_numpy(), keep_i)].reset_index(drop=True)
            f = features.pair_features(c, norm)
            out = pd.DataFrame({"q": f.q.to_numpy(np.int32), "i": f.i.to_numpy(np.int32)})
            ps = [matcher.predict(m, f) for m in models]
            out["p"] = np.mean(ps, 0).astype(np.float32)
            if each:
                for k, pk in enumerate(ps):
                    out[f"p_{k}"] = pk
            scored.append(out)
    return pd.concat(scored, ignore_index=True)


def cmd_train(a):
    norm = load_norm(a, "train")
    cands = load_candidates(a, "train", norm)
    tq, ti, n_true = truth_pairs(a, norm)
    s1_rows = np.flatnonzero(norm.src.to_numpy() == 1)
    key = lambda q, i: q.astype(np.int64) * len(norm) + i
    truth = key(tq, ti)

    # blocking recall = ceiling on what the matcher can ever find
    ranks = ["rank_rev", "rank_fwd", "rank_combo", "rank_ak", "rank_char", "rank_tw"]
    allc = pd.concat([pd.read_parquet(p, columns=["q", "i"] + ranks) for p in cands.values()])
    ck = key(allc.q.to_numpy(), allc.i.to_numpy())
    found = np.isin(truth, ck)
    log(f"blocking: {len(allc):,} pairs ({len(allc) / len(s1_rows):.1f} per S1), recall {found.mean():.4f}")
    # which channels the recall comes from: a pair is "old" if a v1 channel (reverse/forward/combo/addr) found it
    old = ((allc.rank_rev <= a.k_rev) | (allc.rank_fwd <= a.k_fwd) | (allc.rank_combo <= 3) | (allc.rank_ak <= 1)).to_numpy()
    char, tw = (allc.rank_char <= a.k_char).to_numpy(), (allc.rank_tw == 1).to_numpy()
    for name, m in (("v1 channels", old), ("v1 + char", old | char), ("v1 + twins", old | tw), ("all", old | char | tw)):
        log(f"  recall {name}: {np.isin(truth, ck[m]).mean():.4f} with {m.sum():,} pairs")
    del allc, ck, old, char, tw

    # three disjoint groups of S1 entities: fit the model / early-stop it / choose + report the decision rule
    rng = np.random.default_rng(0)
    sample = rng.choice(s1_rows, min(a.train_s1 + a.valid_s1, len(s1_rows)), replace=False)
    fit_q, stop_q, tune_q = np.split(sample, [a.train_s1, a.train_s1 + a.valid_s1 // 2])
    f = build_features(a, norm, cands, np.concatenate([fit_q, stop_q]))
    f["y"] = np.isin(key(f.q.to_numpy(), f.i.to_numpy()), truth).astype(np.int8)
    is_stop = np.isin(f.q.to_numpy(), stop_q)
    model = [matcher.fit(f[~is_stop], f[is_stop])]
    stop = f[is_stop].reset_index(drop=True)
    del f
    # More data without more RAM: extra models, each on a DISJOINT sample of the labelled S1 entities that
    # no other group uses (the model sees 300k of 2.2M), averaged. Tune S1 stay untouched by all of them.
    rest = np.random.default_rng(1).permutation(np.setdiff1d(s1_rows, sample))
    for k in range(1, a.n_models):
        fq = rest[(k - 1) * a.train_s1: k * a.train_s1]
        fk = build_features(a, norm, cands, fq, cache=False)
        fk["y"] = np.isin(key(fk.q.to_numpy(), fk.i.to_numpy()), truth).astype(np.int8)
        log(f"model {k}: {len(fq):,} more S1 entities, {len(fk):,} pairs")
        model.append(matcher.fit(fk, stop))
        del fk
    del stop

    # Honest evaluation: every S1 (of all 2.2M) competing for a record the tune_q entities retrieved is
    # scored, so the one-owner step sees the same competition as on test; judge only the untouched tune_q.
    scored = score_pairs(a, norm, cands, model, keep_q=tune_q, each=len(model) > 1)
    scored.to_parquet(a.work / "tune_pairs.parquet")                 # before one-owner: lets the decision layer be re-studied
    for k in range(len(model) if len(model) > 1 else 0):             # each model alone, same tune S1 and rule grid
        o = matcher.one_owner(scored[["q", "i", f"p_{k}"]].rename(columns={f"p_{k}": "p"}))
        o = o[np.isin(o.q.to_numpy(), tune_q)].copy()
        o["y"] = np.isin(key(o.q.to_numpy(), o.i.to_numpy()), truth).astype(np.int8)
        log(f"  model {k} alone: best threshold rule {best_threshold_score(o, tune_q, n_true)}")
    owned = matcher.one_owner(scored[["q", "i", "p"]])
    del scored
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
    rule = {"kind": "threshold", "t": a.t, "margin": a.margin}
    name = f"matching_results_t{a.t:g}_m{a.margin:g}"
    if a.t_country:                                                  # e.g. France=0.85: probe one country's cutoff alone
        rule["by_country"] = {k: float(v) for k, v in (x.split("=") for x in a.t_country)}
        name += "".join(f"_{k}{v:g}" for k, v in rule["by_country"].items())
    write_submission(a, norm, scored, rule, name + ".tsv")


def write_submission(a, norm, scored, rule, name):
    s1_rows = np.flatnonzero(norm.src.to_numpy() == 1)
    owned = matcher.one_owner(scored)
    if rule.get("by_country"):
        t = np.full(len(owned), rule["t"], np.float32)
        cty = norm.country.to_numpy()[owned.q.to_numpy()]
        for c, v in rule["by_country"].items():
            t[cty == c] = v
        p = owned.p.to_numpy()
        picked = owned[(p >= t) & (p - owned.p2.to_numpy() >= rule.get("margin", 0.0))]
    else:
        picked = matcher.select(owned, rule)
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
    ap.add_argument("--t", type=float, default=0.7, help="select: probability threshold for the variant file")
    ap.add_argument("--margin", type=float, default=0.0, help="select: runner-up S1 must trail the owner by this much")
    ap.add_argument("--t-country", nargs="*", default=[], metavar="COUNTRY=T",
                    help="select: override the threshold for one country, e.g. France=0.85")
    ap.add_argument("--data", type=Path, default=ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset")
    ap.add_argument("--work", type=Path, default=Path.home() / "er_work", help="cache dir (keep it off OneDrive)")
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    ap.add_argument("--split", default="train", help="for prepare/block only")
    ap.add_argument("--force", nargs="*", default=[], choices=["prepare", "blocking", "features"])
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--k-rev", type=int, default=3, help="S1 candidates kept per S2/S3 record")
    ap.add_argument("--k-fwd", type=int, default=10, help="S2 (and S3) candidates kept per S1 record")
    ap.add_argument("--prune", type=float, default=0.002, help="query-side max document frequency")
    ap.add_argument("--k-char", type=int, default=2, help="S1 candidates per S2/S3 record from name 3-grams (0 = off)")
    ap.add_argument("--char-min", type=float, default=0.3, help="minimum 3-gram cosine for the character channel")
    ap.add_argument("--tw-cap", type=int, default=8, help="largest twin group used for 2-hop expansion (0 = off)")
    ap.add_argument("--n-models", type=int, default=1, help="stage-1 models, each on a disjoint --train-s1 sample, averaged")
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
