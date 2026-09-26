"""Candidate generation (blocking) + TF-IDF similarity scores.

Scoring every S1 record against millions of S2/S3 records is impossible, so
we retrieve a short list of plausible candidates first, per country.

Representation: TF-IDF over words AND word pairs (bigrams) of the name and
the address. A bigram like "12029 sheraton" (house number + street) is
extremely rare, so it pins down the right record cheaply; unigrams keep
recall when the number is corrupted. Words are hashed (no vocabulary to hold
in RAM) and very common words are dropped from the query side only, which is
what keeps the sparse search fast.

Retrieval channels, unioned:
  reverse : every S2/S3 record looks up its k most similar S1 records. Each
            S2/S3 record belongs to at most ONE S1 entity and S1 has no
            duplicates, so the owner is usually rank 1 (95% of the time).
  forward : every S1 record looks up its k most similar records per source,
            catching owners that a generic-looking S2/S3 record ranked lower.
  combo   : every S2/S3 record looks up its k best S1 on "house number x name
            sound" keys (see combo_keys). Built for India, where a name often
            arrives in Kannada/Hindi script and ~48 S1 share the same generic
            name: 33% of India's previously missed pairs are found this way.
  addr    : every S2/S3 record looks up its best S1 on "house number x address
            word" keys (see addr_keys). Built for France's dense streets.
  char    : every S2/S3 record looks up its k best S1 on character 3-grams of the
            space-free name (the character channel of Sparkly-style TF-IDF
            blockers and the Foursquare winners). Whole-word search cannot link
            "firstyhn.com" to "First Yhn", or a name with a typo in every word.
  twins   : 2-hop expansion. A record's twins (other S2/S3 records with the same
            address, or the same name sound + house numbers) join every S1 that
            the record is the best match of in some channel: copies of a business
            agree with each other, so a garbled copy rides along with a clean one.
Every channel leaves a rank_* column (k+1 = "not retrieved this way"), which is
both a model feature and how the train log attributes recall to channels.
Measured on train (US): 97.9% of true pairs survive, ~20 candidates per S1.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

VERSION = 2          # bump when candidate generation changes: cached candidate files carry it in their name

HV = HashingVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), n_features=2 ** 24,
                       alternate_sign=False, norm=None, dtype=np.float32)


HV1 = HashingVectorizer(token_pattern=r"\S+", ngram_range=(1, 1), n_features=2 ** 24,   # keys: no bigrams, their order is arbitrary
                        alternate_sign=False, norm=None, dtype=np.float32)
HVC = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 22, lowercase=False,
                        alternate_sign=False, norm=None, dtype=np.float32)      # "firstyhn" -> " fi", "fir", ..., "hn "


def _hash(texts):
    return HV.transform(texts)


def _hash_keys(texts):
    return HV1.transform(texts)


def _hash_chars(texts):
    return HVC.transform(texts)


def hashed_counts(texts, pool, fn=_hash):
    chunks = [texts[s:s + 200_000] for s in range(0, len(texts), 200_000)]
    return sp.vstack(pool.map(fn, chunks)).tocsr()


def tfidf(X):
    return TfidfTransformer(sublinear_tf=True).fit_transform(X).astype(np.float32).tocsr()


def prune_common(X, max_df_frac):
    """Drop very common columns from the *query* side. They barely move a
    cosine score but dominate the cost of the sparse product."""
    df = np.bincount(X.indices, minlength=X.shape[1])
    keep = df <= max(max_df_frac * X.shape[0], 50)                 # floor: a small country must keep its terms
    Y = (X @ sp.diags(keep.astype(np.float32))).tocsr()
    Y.eliminate_zeros()
    return Y


def topk(Q, I, k, threshold=0.05, chunk=100_000):
    """For each row of Q: the k most cosine-similar rows of I -> (q, i, rank)."""
    IT = I.T.tocsr()
    qs, is_, rs = [], [], []
    for s in range(0, Q.shape[0], chunk):
        M = sp_matmul_topn(Q[s:s + chunk], IT, top_n=k, threshold=threshold, sort=True, n_threads=8).tocsr()
        n = np.diff(M.indptr)
        qs.append(np.repeat(np.arange(s, s + M.shape[0]), n))
        is_.append(M.indices)
        rs.append(np.arange(M.nnz) - np.repeat(M.indptr[:-1], n) + 1)   # 1 = best
    return np.concatenate(qs), np.concatenate(is_), np.concatenate(rs).astype(np.float32)


def rowdot(X, a, b, chunk=2_000_000):
    """cosine(X[a[j]], X[b[j]]) for every j (rows are L2-normalised)."""
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.asarray(X[a[s:s + chunk]].multiply(X[b[s:s + chunk]]).sum(1)).ravel()
    return out


def unmatched_mass(X, a, b, chunk=2_000_000):
    """Share of X[a]'s TF-IDF weight on terms that X[b] does not contain.
    High on both sides = each name has a rare word the other lacks
    ("Luciani Markel *Excavation*" vs "Luciani Markel *Hospitality*")."""
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), chunk):
        A, B = X[a[s:s + chunk]], X[b[s:s + chunk]].copy()
        B.data[:] = 1
        out[s:s + chunk] = 1 - np.asarray(A.multiply(A).multiply(B).sum(1)).ravel()
    return out


def combo_keys(phon, nums):
    """'house number x name sound' keys: '79_sftvr' = number 79 + skeleton of 'software'.
    Kannada 'saaphttveer' and English 'software' share the skeleton, so the key links a
    transliterated record to its S1 even when the words differ, and a number+name pair is rare."""
    return [" ".join(f"{n}_{p}" for n in ns.split() for p in ps.split()) for ps, ns in zip(phon, nums)]


def addr_keys(addr, nums):
    """'house number x address word' keys: '27@montesquieu'. A French street holds ~14 businesses
    and house numbers repeat, so '27' and the street words alone are too common to survive query
    pruning: 23% of France's same-address, different-name copies were never retrieved (US/India:
    0%). Joined to the number they are rare, whatever the name says. Own channel, so these keys
    never crowd the name-sound keys out of the combo top-k (India has ~4x more of them)."""
    return [" ".join(f"{n}@{w}" for n in ns.split() for w in a.split() if not w.isdigit()) for a, ns in zip(addr, nums)]


def twin_pairs(keys, pl, cap):
    """(i, j) for every two DIFFERENT S2/S3 records sharing a key; keys < 0 = no key.
    Groups larger than cap are skipped: a key that common is not a twin signal."""
    k = keys[pl]
    ok = k >= 0
    g = pd.DataFrame({"r": pl[ok], "k": k[ok]})
    size = g.groupby("k").r.transform("size")
    g = g[(size >= 2) & (size <= cap)]
    m = g.merge(g, on="k", suffixes=("", "_j"))
    m = m[m.r != m.r_j]
    return m.r.to_numpy(), m.r_j.to_numpy()


def twin_expand(c, d, cap=8):
    """New (q, j) pairs: j is a twin of a record i whose best S1 in some channel is q.
    Returns them with rank_tw = 1; pairs already in c get rank_tw = 1 too when they would
    have been added (so recall can be attributed), else 2."""
    src = d.src.to_numpy()
    pl = np.flatnonzero(src != 1)
    has_a, has_n = d.addr.str.len().to_numpy() > 0, d.nums.str.len().to_numpy() > 0
    ka = np.where(has_a, pd.factorize(d.addr.to_numpy())[0], -1)
    ks = np.where(has_n, pd.factorize((d.phon + "|" + d.nums).to_numpy())[0], -1)
    strong = c.loc[(c.rank_rev == 1) | (c.rank_combo == 1) | (c.rank_ak == 1) | (c.rank_char == 1), ["q", "i"]]
    new = []
    for keys in (ka, ks):
        i, j = twin_pairs(keys, pl, cap)
        t = pd.DataFrame({"i": i, "j": j}).merge(strong, on="i")
        new.append(t[["q", "j"]])
    new = pd.concat(new, ignore_index=True).drop_duplicates().rename(columns={"j": "i"})
    return new


def block_country(d, k_rev, k_fwd, prune, pool, k_combo=3, k_ak=1, k_char=2, char_min=0.3, tw_cap=8):
    """d: normalised records of ONE country (all sources). Returns candidate
    pairs (q = S1 position, i = S2/S3 position in d) with TF-IDF features.
    k_char=0 / tw_cap=0 switch the character / twin channels off (the v1 candidate set)."""
    t = time.time()
    src = d.src.to_numpy()
    s1, pl = np.flatnonzero(src == 1), np.flatnonzero(src != 1)
    if not len(s1) or not len(pl):
        return None
    A = tfidf(hashed_counts(addr_keys(d.addr.tolist(), d.nums.tolist()), pool, _hash_keys))
    q, i, r = topk(prune_common(A[pl], prune), A[s1], k_ak)
    parts = [pd.DataFrame({"q": s1[i], "i": pl[q], "rank_ak": r})]
    del A
    K = tfidf(hashed_counts(combo_keys(d.phon.tolist(), d.nums.tolist()), pool, _hash_keys))
    q, i, r = topk(prune_common(K[pl], prune), K[s1], k_combo)
    parts.append(pd.DataFrame({"q": s1[i], "i": pl[q], "rank_combo": r}))

    C = tfidf(hashed_counts(d.nosp.tolist(), pool, _hash_chars))
    if k_char:
        q, i, r = topk(prune_common(C[pl], prune), C[s1], k_char, threshold=char_min)
        parts.append(pd.DataFrame({"q": s1[i], "i": pl[q], "rank_char": r}))
        log_n = len(q)

    names = (d.core + " " + d.alt).str.strip().tolist()
    addrs = [" ".join("@" + w for w in s.split()) for s in d.addr.tolist()]   # "@": address terms never collide with name terms
    Xn, Xa = hashed_counts(names, pool), hashed_counts(addrs, pool)
    W, Nw, Aw = tfidf(Xn + Xa), tfidf(Xn), tfidf(Xa)
    del Xn, Xa, names, addrs

    q, i, r = topk(prune_common(W[pl], prune), W[s1], k_rev)
    parts.append(pd.DataFrame({"q": s1[i], "i": pl[q], "rank_rev": r}))
    Wq = prune_common(W[s1], prune)
    for s in (2, 3):
        ps = np.flatnonzero(src == s)
        if len(ps):
            q, i, r = topk(Wq, W[ps], k_fwd)
            parts.append(pd.DataFrame({"q": s1[q], "i": ps[i], "rank_fwd": r}))
    c = pd.concat(parts, ignore_index=True).groupby(["q", "i"], sort=False).min().reset_index()
    for col, k in (("rank_rev", k_rev), ("rank_fwd", k_fwd), ("rank_combo", k_combo), ("rank_ak", k_ak), ("rank_char", k_char)):
        c[col] = c[col].fillna(k + 1) if col in c else np.float32(k + 1)
    if tw_cap:
        tw = twin_expand(c, d, tw_cap)
        tw["rank_tw"] = np.float32(1)
        n0 = len(c)
        c = c.merge(tw, on=["q", "i"], how="outer")
        c["rank_tw"] = c.rank_tw.fillna(2).astype(np.float32)
        for col, k in (("rank_rev", k_rev), ("rank_fwd", k_fwd), ("rank_combo", k_combo), ("rank_ak", k_ak), ("rank_char", k_char)):
            c[col] = c[col].fillna(k + 1).astype(np.float32)
        print(f"    twins: +{len(c) - n0:,} pairs", flush=True)
    else:
        c["rank_tw"] = np.float32(2)
    if k_char:
        print(f"    char channel: {log_n:,} retrievals", flush=True)
    q, i = c.q.to_numpy(), c.i.to_numpy()
    c["src"] = src[i].astype(np.int8)
    c["cos_k"] = rowdot(K, q, i)
    del K
    c["cos_c"] = rowdot(C, q, i)
    del C
    c["cos_w"] = rowdot(W, q, i)
    c["cos_nw"] = rowdot(Nw, q, i)
    c["cos_aw"] = rowdot(Aw, q, i)
    c["unm_a"] = unmatched_mass(Nw, q, i)
    c["unm_b"] = unmatched_mass(Nw, i, q)
    del W, Nw, Aw, Wq                                               # free the matrices before the pandas-heavy part
    add_context(c)
    print(f"    {len(s1):,} S1 x {len(pl):,} S2/S3 -> {len(c):,} pairs "
          f"({len(c) / len(s1):.1f}/S1) in {time.time() - t:.0f}s", flush=True)
    return c


def add_context(c):
    """Where does this pair stand among its competitors?"""
    for col in ("cos_w", "cos_nw", "cos_aw", "cos_k", "cos_c"):
        g = c.groupby(["q", "src"])[col]
        c[f"{col}_gap"] = g.transform("max") - c[col]              # distance to this S1's best candidate
        c[f"{col}_rk"] = g.rank(ascending=False, method="min").astype(np.float32)
        r = c.groupby("i")[col]
        c[f"{col}_rgap"] = r.transform("max") - c[col]             # distance to the candidate's best S1
    c["n_s1_for_i"] = c.groupby("i").q.transform("size").astype(np.float32)
    c["n_high"] = (c.cos_w > 0.5).groupby([c.q, c.src]).transform("sum").astype(np.float32)   # no frame copy (OOM at 45M rows)
    add_margins(c)


MARGIN_COLS = [f"{c}_{s}" for c in ("cos_w", "cos_nw", "cos_aw") for s in ("rmarg", "qmarg")] + ["n_close_i"]


def best_other(g, v):
    """For rows grouped by g: the best value among the OTHER rows of the same group (-1 if alone)."""
    o = np.lexsort((-v, g))
    gs, vs = g[o], v[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    nxt = np.minimum(start + 1, len(gs) - 1)
    top2 = np.where((gs[nxt] == gs) & (nxt != start), vs[nxt], -1.0)
    out = np.empty(len(v), np.float32)
    out[o] = np.where(np.arange(len(gs)) == start, top2, vs[start])
    return out


def add_margins(c):
    """Self minus the best OTHER claimant. *_rgap is 0 for the winner whether it wins by a mile or ties
    (two S1 entities with the same name and an empty-address candidate); *_rmarg tells them apart."""
    q, i = c.q.to_numpy(), c.i.to_numpy()
    qs = q.astype(np.int64) * 4 + c.src.to_numpy()
    for col in ("cos_w", "cos_nw", "cos_aw"):
        v = c[col].to_numpy().astype(np.float32)
        bo = best_other(i, v)
        c[f"{col}_rmarg"] = np.where(bo >= 0, v - bo, np.nan).astype(np.float32)   # vs other S1s claiming this record
        bq = best_other(qs, v)
        c[f"{col}_qmarg"] = np.where(bq >= 0, v - bq, np.nan).astype(np.float32)   # vs this S1's other candidates
    v = c.cos_w.to_numpy().astype(np.float32)
    top = pd.Series(v).groupby(i).transform("max").to_numpy()
    c["n_close_i"] = pd.Series(v >= top - 0.05).groupby(i).transform("sum").to_numpy().astype(np.float32)


def demo():
    from multiprocessing.dummy import Pool as ThreadPool
    import normalize
    recs = [(1, "First Yhn", "12 Main St, Springfield, IL"), (1, "Kashvi Electricals", "9 MG Road, Pune"),
            (1, "Oak Bakery", "40 Oak Ave, Dover, DE"),
            (2, "firstyhn.com", ""), (3, "Kashvii Electrcals", ""), (2, "Oak Bakery", "40 Oak Ave, Dover, DE"),
            (3, "Qzvntrola", "40 Oak Ave, Dover, DE")]
    rows = [(s,) + normalize.norm_name(n) + normalize.norm_addr(a) for s, n, a in recs]
    d = pd.DataFrame(rows, columns=["src", "core", "alt", "legal", "phon", "nosp", "nl", "addr", "nums"])
    with ThreadPool(2) as pool:
        c = block_country(d, 3, 10, 1.0, pool, char_min=0.3)
    got = {(int(a), int(b)): r for a, b, r in zip(c.q, c.i, c.rank_char)}
    assert got.get((0, 3)) == 1, got                               # website name found by 3-grams
    assert got.get((1, 4)) == 1, got                               # typo in every word found by 3-grams
    tw = {(int(a), int(b)): r for a, b, r in zip(c.q, c.i, c.rank_tw)}
    assert tw.get((2, 6)) == 1 and tw.get((2, 5)) == 1, tw         # garbled copy joins its clean twin's S1
    # a pair no channel found: record 5 is a twin (same address) of record 3, whose best S1 is 0
    c = pd.DataFrame({"q": [0], "i": [3], "rank_rev": [1.0], "rank_combo": [4.0], "rank_ak": [2.0], "rank_char": [3.0]})
    d = pd.DataFrame({"src": [1, 2, 2, 2, 3, 3], "addr": ["a", "x", "", "y", "", "y"], "nums": ["", "", "", "", "", ""],
                      "phon": [""] * 6})
    new = twin_expand(c, d.iloc[[0, 1, 2, 3, 4, 5]].reset_index(drop=True))
    assert sorted(map(tuple, new[["q", "i"]].to_numpy().tolist())) == [(0, 5)], new
    print("blocking.demo OK")


if __name__ == "__main__":
    demo()
