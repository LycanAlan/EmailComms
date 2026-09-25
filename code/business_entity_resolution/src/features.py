"""Pair features: how alike are an S1 record and one of its candidates?

Every feature is a country-agnostic *similarity*, never a country flag, so the
model transfers to France (absent from training). String similarities come
from rapidfuzz.process.cpdist, which scores list-vs-list element-wise in C++
on all cores, so tens of millions of pairs take minutes.

Feature groups
  name     : edit-distance / token-set / Jaro-Winkler on the cleaned name, its
             phonetic skeleton, its no-space form, and alias names
  legal    : does either side carry a legal form, do the families agree/conflict
  address  : fuzzy address similarity + house-number logic. The decoys in this
             data are the same business name with a *nearby* house number
             (20999 vs 20996), while true copies drop digits (6100 -> 100); the
             number features separate the two. (Zero padding, 357 -> 00357, is
             already undone in normalize.py.)
  context  : TF-IDF cosines and ranks from blocking (computed in blocking.py)

A side with no text (empty address, name that was only a legal form) gives NaN,
not 0 or 100: "unknown" is different from "dissimilar", and LightGBM learns
what to do with missing values.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import fingerprints

NAN = np.float32(np.nan)
VERSION = 2          # bump when the feature set changes: cached feature tables carry it in their file name


def _sim(a, b, scorer):
    s = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    s[[not x or not y for x, y in zip(a, b)]] = NAN               # missing, not dissimilar
    return s


def number_features(na, nb):
    """na, nb: space-separated number strings (house/plot/zip...) of each side."""
    out = np.full((len(na), 8), NAN, np.float32)
    for j, (x, y) in enumerate(zip(na, nb)):
        if not x or not y:
            continue                                               # missing side -> NaN, LightGBM handles it
        A, B = x.split(), y.split()
        sa, sb = set(A), set(B)
        common = len(sa & sb)
        ua, ub = sa - sb, sb - sa
        trunc, diff = 0.0, None
        for u in ua:                                               # compare the numbers that did NOT match
            for v in ub:
                if len(u) < 2 or len(v) < 2:                       # skip "1/2", unit letters split off, ...
                    continue
                if u.endswith(v) or v.endswith(u) or u.startswith(v) or v.startswith(u):
                    trunc = 1.0                                    # 6100 vs 100: dropped digit -> typical noise
                if len(u) < 10 and len(v) < 10:
                    diff = min(diff, abs(int(u) - int(v))) if diff is not None else abs(int(u) - int(v))
        out[j] = (A[0] == B[0], common, common / len(sa | sb), len(ua), len(ub), trunc,
                  diff is not None and diff <= 20,                 # 20999 vs 20996: typical decoy
                  np.log1p(diff) if diff is not None else NAN)
    return out


def legal_features(la, lb):
    out = np.zeros((len(la), 4), np.float32)
    for j, (x, y) in enumerate(zip(la, lb)):
        fa, fb = set(x.split()) - {"co"}, set(y.split()) - {"co"}  # "Co"/"Cie" is too generic to conflict
        out[j] = (bool(x), bool(y), bool(fa & fb), bool(fa and fb and not fa & fb))
    return out


def pair_features(c, norm):
    """c: candidate pairs (q, i + blocking scores). Returns c with feature columns added."""
    q, i = c.q.to_numpy(), c.i.to_numpy()
    A = lambda col: norm[col].take(q).tolist()
    B = lambda col: norm[col].take(i).tolist()
    f = c.copy()

    ca, cb = A("core"), B("core")
    f["n_ratio"] = _sim(ca, cb, fuzz.ratio)
    f["n_tset"] = _sim(ca, cb, fuzz.token_set_ratio)
    f["n_tsort"] = _sim(ca, cb, fuzz.token_sort_ratio)
    f["n_partial"] = _sim(ca, cb, fuzz.partial_ratio)
    f["n_jw"] = _sim(ca, cb, JaroWinkler.normalized_similarity)
    f["n_first_eq"] = np.array([x.split()[0] == y.split()[0] if x and y else NAN for x, y in zip(ca, cb)], np.float32)
    f["n_len_a"] = np.array([x.count(" ") + bool(x) for x in ca], np.float32)
    f["n_len_b"] = np.array([x.count(" ") + bool(x) for x in cb], np.float32)
    aa, ab = A("alt"), B("alt")
    f["n_alias"] = np.fmax(np.fmax(_sim(ca, ab, fuzz.token_set_ratio),
                                   _sim(aa, cb, fuzz.token_set_ratio)),
                           _sim(aa, ab, fuzz.token_set_ratio))
    del ca, cb, aa, ab
    na, nb = A("nosp"), B("nosp")
    f["n_nosp"] = _sim(na, nb, fuzz.ratio)
    f["n_nosp_partial"] = _sim(na, nb, fuzz.partial_ratio)
    pa, pb = A("phon"), B("phon")
    f["n_phon"] = _sim(pa, pb, fuzz.ratio)
    f["n_phon_tset"] = _sim(pa, pb, fuzz.token_set_ratio)
    del na, nb, pa, pb
    f["nl_a"] = norm.nl.to_numpy()[q]
    f["nl_b"] = norm.nl.to_numpy()[i]

    lg = legal_features(A("legal"), B("legal"))
    for k, name in enumerate(["leg_a", "leg_b", "leg_same", "leg_conflict"]):
        f[name] = lg[:, k]

    da, db = A("addr"), B("addr")
    f["a_tsort"] = _sim(da, db, fuzz.token_sort_ratio)
    f["a_tset"] = _sim(da, db, fuzz.token_set_ratio)
    f["a_partial"] = _sim(da, db, fuzz.partial_ratio)
    f["a_empty_b"] = np.array([not y for y in db], np.float32)
    del da, db
    ua, ub = A("nums"), B("nums")
    nf = number_features(ua, ub)
    for k, name in enumerate(["num_first_eq", "num_common", "num_jacc", "num_only_a", "num_only_b",
                              "num_trunc", "num_near", "num_logdiff"]):
        f[name] = nf[:, k]
    f["num_digits"] = _sim([x.replace(" ", "") for x in ua], [y.replace(" ", "") for y in ub], fuzz.ratio)
    return fingerprints.pair_extras(f, norm, q, i)


def demo():
    nf = number_features(["20996 2556", "6100", "357 1 2", ""], ["20999 2556", "100", "357", "5"])
    assert nf[0, 1] == 1 and nf[0, 6] == 1, nf[0]                # decoy: shared 2556, near-miss 20996/20999
    assert nf[1, 5] == 1 and nf[1, 6] == 0, nf[1]                 # truncation 6100 -> 100
    assert nf[2, 0] == 1 and nf[2, 5] == 0 and nf[2, 6] == 0     # house number exact
    assert np.isnan(nf[3]).all()                                  # missing side
    nf = number_features(["1 100"], ["2 100"])
    assert nf[0, 5] == 0 and nf[0, 6] == 0, nf[0]                 # single digits ("1/2") never count as near/trunc
    s = _sim(["abc", "", "x"], ["abc", "", ""], fuzz.ratio)
    assert s[0] == 100 and np.isnan(s[1]) and np.isnan(s[2])      # empty -> NaN, never a fake 100
    lg = legal_features(["llp", "ltd", "", "co inc"], ["ltd", "ltd", "llc", "inc"])
    assert lg[0, 3] == 1 and lg[1, 2] == 1 and lg[2, 3] == 0 and lg[3, 2] == 1
    print("features.demo OK")


if __name__ == "__main__":
    demo()
