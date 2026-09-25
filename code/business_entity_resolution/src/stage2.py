"""Stage 2: re-score each record's owner pick using what stage 1 said about RELATED pairs.

Stage 1 scores one (S1, record) pair at a time. Stage 2 sees, for the owned pair (q, i): the
runner-up S1's probability for i, how many confident copies q already has, and whether i's
"twins" (other S2/S3 records with the same address, or the same name sound + house numbers)
are confident copies of q. A true business has several copies that agree with each other; a
decoy is a one-off. Trained on the tune S1 entities' owned pairs (their stage-1 scores are
honest: those entities never trained stage 1); 2-fold out-of-fold on v4's tune set: 0.9796 -> 0.9808.

  python src/stage2.py fit   --work W            # OOF check on W/tune_scored.parquet, saves W/stage2/
  python src/stage2.py apply --work W --out O    # W/test_scored.parquet -> O/matching_results.tsv
"""
import lightgbm as lgb  # before sklearn (Windows)
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import matcher

FEATS = ["p", "p2", "m", "n_q", "n_conf_q", "sum_p_q", "rank_q", "max_other_q", "tw_a", "tw_s", "tw_a_maxp", "tw_s_maxp"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
              num_threads=8, verbose=-1, seed=0)
ROUNDS = 400


def twin_keys(norm_path):
    """Per record: address key and sound+numbers key (per country; -1/-2 when empty) and the number
    of OTHER S2/S3 records sharing each."""
    f = pq.ParquetFile(norm_path)
    col = lambda c: f.read(columns=[c])[c]
    L = lambda x: pc.cast(x, pa.large_string())
    def key(*cols):
        s = L(col("country"))
        for c in cols:
            s = pc.binary_join_element_wise(s, L(col(c)), pa.scalar("|", pa.large_string()))
        return pc.dictionary_encode(s.combine_chunks()).indices.to_numpy()
    pl = col("src").to_numpy() != 1
    has_a = pc.greater(pc.utf8_length(col("addr")), 0).to_numpy(zero_copy_only=False)
    has_n = pc.greater(pc.utf8_length(col("nums")), 0).to_numpy(zero_copy_only=False)
    ka, ks = key("addr"), key("phon", "nums")
    tw_a = np.where(has_a, np.bincount(ka[pl], minlength=ka.max() + 1)[ka] - pl, -1)
    tw_s = np.where(has_n, np.bincount(ks[pl], minlength=ks.max() + 1)[ks] - pl, -1)
    return np.where(has_a, ka, -1), np.where(has_n, ks, -2), tw_a.astype(np.float32), tw_s.astype(np.float32)


def features(own, keys):
    """own: one_owner() output (q, i, p, p2). Adds the stage-2 columns in place."""
    ka, ks, tw_a, tw_s = keys
    i = own.i.to_numpy()
    own["ka"], own["ks"], own["tw_a"], own["tw_s"] = ka[i], ks[i], tw_a[i], tw_s[i]
    own["m"] = own.p - own.p2
    g = own.groupby("q").p
    own["n_q"] = g.transform("size").astype(np.float32)
    own["n_conf_q"] = (own.p >= 0.9).groupby(own.q).transform("sum").astype(np.float32)
    own["sum_p_q"] = g.transform("sum")
    own["rank_q"] = g.rank(ascending=False, method="min")
    own["max_other_q"] = (g.transform("max") - own.p).clip(lower=0)
    for k, name in (("ka", "tw_a_maxp"), ("ks", "tw_s_maxp")):          # best p among i's OTHER twins owned by q
        s = own.sort_values("p", ascending=False, kind="stable")
        grp = s.groupby(["q", k])
        r = grp.cumcount().to_numpy()
        first = grp.p.transform("first").to_numpy()
        second = s.p.where(r == 1).groupby([s.q, s[k]]).transform("max").to_numpy()
        best = np.where(r == 0, second, first)
        best[s[k].to_numpy() < 0] = np.nan                               # empty address / no numbers: no twins
        own[name] = pd.Series(best, index=s.index).reindex(own.index)
    return own


def fit_model(d):
    return lgb.train(PARAMS, lgb.Dataset(d[FEATS].to_numpy(np.float32), d.y.to_numpy(np.float32)), ROUNDS)


def tune_rows_and_truth(work, data, train_s1=300_000, valid_s1=100_000):
    """The pipeline's tune S1 rows (same seed/split as cmd_train) and n_true per S1 row."""
    ids = pq.read_table(work / "train_norm.parquet", columns=["id"])["id"].to_numpy(zero_copy_only=False)
    src = pq.read_table(work / "train_norm.parquet", columns=["src"])["src"].to_numpy()
    s1_rows = np.flatnonzero(src == 1)
    sample = np.random.default_rng(0).choice(s1_rows, min(train_s1 + valid_s1, len(s1_rows)), replace=False)
    tune_q = np.sort(sample[train_s1 + valid_s1 // 2:])
    pos = pd.Series(np.arange(len(ids)), index=ids)
    gt = pd.read_csv(data / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    n = (gt.matched_entity_ids.str.len().gt(0) * (gt.matched_entity_ids.str.count(",") + 1)).to_numpy()
    return tune_q, pd.Series(n, index=pos[gt.source1_entity_id].to_numpy())


def cmd_fit(a):
    ev = pd.read_parquet(a.work / "tune_scored.parquet").reset_index(drop=True)
    features(ev, twin_keys(a.work / "train_norm.parquet"))
    tune_q, n_true = tune_rows_and_truth(a.work, a.data, a.train_s1, a.valid_s1)
    fold = (pd.util.hash_array(ev.q.to_numpy().astype(np.int64)) % 2).astype(int)
    ev["p_s2"] = np.nan
    for k in (0, 1):
        ev.loc[fold == k, "p_s2"] = fit_model(ev[fold != k]).predict(ev.loc[fold == k, FEATS].to_numpy(np.float32))
    score = lambda col, t, m=0.0: matcher.macro_f05(ev[(ev[col] >= t) & (ev.p - ev.p2 >= m)], tune_q, n_true)
    s1 = max((score("p", t, m), t, m) for t in np.arange(0.4, 0.96, 0.025) for m in (0.0, 0.2, 0.4, 0.6))
    s2 = max((score("p_s2", t), t) for t in np.arange(0.3, 0.96, 0.025))
    print(f"stage 1 alone: {s1[0]:.4f} (t={s1[1]:.3f}, margin={s1[2]}) | stage 2 out-of-fold: {s2[0]:.4f} (t={s2[1]:.3f})", flush=True)
    out = a.work / "stage2"
    out.mkdir(exist_ok=True)
    fit_model(ev).save_model(str(out / "model.txt"))
    json.dump({"t": float(s2[1]), "oof": float(s2[0]), "stage1": float(s1[0]), "use": bool(s2[0] > s1[0])}, open(out / "rule.json", "w"))
    print(f"saved {out}", flush=True)


def cmd_apply(a):
    import pipeline                                                  # write_lists + the S1 order of the test split
    rule = json.load(open(a.work / "stage2" / "rule.json"))
    if not rule["use"]:
        print("stage 2 did not beat stage 1 out-of-fold; not applying", flush=True)
        return
    own = matcher.one_owner(pd.read_parquet(a.work / "test_scored.parquet"))
    features(own, twin_keys(a.work / "test_norm.parquet"))
    own["p_s2"] = lgb.Booster(model_file=str(a.work / "stage2" / "model.txt")).predict(own[FEATS].to_numpy(np.float32))
    picked = own[own.p_s2 >= rule["t"]]
    t = pq.read_table(a.work / "test_norm.parquet", columns=["id", "src"])
    ids, s1_rows = t["id"].to_numpy(zero_copy_only=False), np.flatnonzero(t["src"].to_numpy() == 1)
    a.out.mkdir(parents=True, exist_ok=True)
    pipeline.write_lists(a.out / "matching_results.tsv", "matched_entity_ids", ids, s1_rows, picked.q.to_numpy(), picked.i.to_numpy())
    print(f"wrote {a.out / 'matching_results.tsv'}: {len(picked):,} matches, stage-2 threshold {rule['t']:.3f}", flush=True)


def demo():
    own = pd.DataFrame({"q": [1, 1, 1, 2], "i": [10, 11, 12, 13], "p": [0.99, 0.6, 0.2, 0.9], "p2": [0.0, 0.1, 0.0, 0.0]})
    ka = np.array([-1] * 10 + [5, 5, 7, -1]); ks = np.array([-2] * 14)
    features(own, (ka, ks, np.zeros(14, np.float32), np.zeros(14, np.float32)))
    assert own.tw_a_maxp.tolist()[:2] == [0.6, 0.99] and np.isnan(own.tw_a_maxp[2]) and np.isnan(own.tw_a_maxp[3])
    assert np.isnan(own.tw_s_maxp).all()                                  # no numbers anywhere: no sound twins
    assert own.n_conf_q.tolist() == [1, 1, 1, 1] and own.rank_q.tolist() == [1, 2, 3, 1]
    print("stage2.demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "apply", "demo"])
    ap.add_argument("--work", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data", type=Path, help="dataset root with train/ and test/ (fit)")
    ap.add_argument("--train-s1", type=int, default=300_000, help="fit: same as the train run")
    ap.add_argument("--valid-s1", type=int, default=100_000, help="fit: same as the train run")
    a = ap.parse_args()
    {"fit": cmd_fit, "apply": cmd_apply, "demo": lambda _: demo()}[a.command](a)
