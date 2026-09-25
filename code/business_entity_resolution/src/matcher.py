"""LightGBM pair classifier + the F0.5-aware decision rule.

The classifier turns each candidate pair's features into P(same business).
The decision rule turns probabilities into final lists; this is where F0.5 is
won or lost:

 1. One owner per record. In the training labels every S2/S3 record belongs to
    at most one S1 entity, so a candidate is kept only for the S1 entity that
    gave it the highest probability.
 2. How many to keep per S1 entity. Since F0.5 = 1.25*TP / (0.25*|truth| + k)
    when k records are predicted, plugging in expected values (TP ~ sum of the
    kept probabilities, |truth| ~ sum of all probabilities) scores every k;
    k = 0 ("singleton") scores the chance that nothing matches. We keep the
    best k. A plain probability threshold is also tried; the rule that scores
    best on held-out S1 entities, using the exact competition metric, wins.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

NOT_FEATURES = {"q", "i", "y", "p"}
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=8, verbose=-1, seed=0)


def fit(tr, va, rounds=2000):
    cols = [c for c in tr.columns if c not in NOT_FEATURES]
    xy = lambda f: (f[cols].to_numpy(np.float32), f.y.to_numpy(np.float32))   # plain arrays: LightGBM's C layer chokes on int8 labels
    dtr = lgb.Dataset(*xy(tr), feature_name=cols)
    dva = lgb.Dataset(*xy(va), feature_name=cols, reference=dtr)
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    gain = pd.Series(m.feature_importance("gain"), cols)
    print("feature importance (share of gain):\n", (gain / gain.sum()).sort_values(ascending=False).round(4).head(25))
    return m


def predict(m, f):
    return m.predict(f[m.feature_name()].to_numpy(np.float32), num_threads=8).astype(np.float32)


def one_owner(scored):
    """Step 1: each S2/S3 record stays only with the S1 entity that scored it highest.
    Must see ALL S1 entities at once, or the competition it resolves is missing."""
    return scored.sort_values("p", ascending=False, kind="stable").drop_duplicates("i")


def select(s, rule):
    """Step 2, per S1 entity, on one_owner() output: which candidates to keep."""
    if rule["kind"] == "threshold":
        return s[s.p >= rule["t"]]
    # expected-F0.5 top-k
    s = s.sort_values(["q", "p"], ascending=[True, False], kind="stable")
    g = s.groupby("q", sort=False).p
    expected_truth = g.transform("sum").to_numpy()
    k = g.cumcount().to_numpy() + 1
    f_k = 1.25 * g.cumsum().to_numpy() / (0.25 * expected_truth + k)
    f_k[s.p.to_numpy() < rule["floor"]] = -1                        # never keep very unlikely records
    log_none = np.log1p(-s.p.clip(upper=1 - 1e-6))                    # P(nothing matches) = prod(1 - p)
    f_0 = np.exp(log_none.groupby(s.q.to_numpy()).transform("sum").to_numpy())
    best = pd.Series(f_k).groupby(s.q.to_numpy()).transform("max").to_numpy()
    k_best = pd.Series(np.where(f_k == best, k, 0)).groupby(s.q.to_numpy()).transform("max").to_numpy()
    return s[(k <= k_best) & (best > f_0)]


def macro_f05(picked, s1_rows, n_true):
    """Exact competition metric over the given S1 rows. picked needs q and y."""
    tp = picked.groupby("q").y.sum().reindex(s1_rows, fill_value=0).to_numpy()
    k = picked.groupby("q").size().reindex(s1_rows, fill_value=0).to_numpy()
    g = n_true.reindex(s1_rows).to_numpy()
    f = np.where((g == 0) & (k == 0), 1.0,
                 np.where((g == 0) | (k == 0), 0.0, 1.25 * tp / np.maximum(0.25 * g + k, 1e-9)))
    return f.mean()


def tune_decision(owned, s1_rows, n_true):
    """owned: one_owner() output restricted to s1_rows, with labels y."""
    rules = [{"kind": "threshold", "t": round(t, 3)} for t in np.arange(0.2, 0.96, 0.025).tolist()]
    rules += [{"kind": "expected_f", "floor": fl} for fl in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    best = None
    for r in rules:
        score = macro_f05(select(owned, r), s1_rows, n_true)
        print(f"  {r} -> macro F0.5 {score:.4f}")
        if best is None or score > best[0]:
            best = (score, r)
    print(f"BEST {best[1]} -> {best[0]:.4f}")
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in best[1].items()}


def save(model, rule, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path / "lgb.txt"))
    (path / "rule.json").write_text(json.dumps(rule))


def load(path):
    return lgb.Booster(model_file=str(path / "lgb.txt")), json.loads((path / "rule.json").read_text())


def demo():
    # S1 #1 owns records 10, 11 (true) and weakly 12; S1 #2 wants 10 too but less; S1 #3 has only noise.
    s = pd.DataFrame({"q": [1, 1, 1, 2, 3], "i": [10, 11, 12, 10, 13],
                      "p": [0.95, 0.9, 0.2, 0.6, 0.1], "y": [1, 1, 0, 0, 0]})
    n_true = pd.Series({1: 2, 2: 0, 3: 0})
    picked = select(one_owner(s), {"kind": "expected_f", "floor": 0.0})
    assert sorted(picked.i) == [10, 11], picked                     # 12 too weak, 10 not given to #2
    assert macro_f05(picked, [1, 2, 3], n_true) == 1.0
    assert abs(macro_f05(s.iloc[:3], [1], n_true) - 1.25 * 2 / (0.5 + 3)) < 1e-9   # the PDF's 0.714 example
    print("matcher.demo OK")


if __name__ == "__main__":
    demo()
