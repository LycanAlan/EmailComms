"""Candidate generation (blocking) + TF-IDF similarity scores.

Scoring every S1 record against millions of S2/S3 records is impossible, so
we retrieve a short list of plausible candidates first, per country.

Representation: TF-IDF over words AND word pairs (bigrams) of the name and
the address. A bigram like "12029 sheraton" (house number + street) is
extremely rare, so it pins down the right record cheaply; unigrams keep
recall when the number is corrupted. Words are hashed (no vocabulary to hold
in RAM) and very common words are dropped from the query side only, which is
what keeps the sparse search fast.

Two retrieval directions, unioned:
  reverse : every S2/S3 record looks up its k most similar S1 records. Each
            S2/S3 record belongs to at most ONE S1 entity and S1 has no
            duplicates, so the owner is usually rank 1 (95% of the time).
  forward : every S1 record looks up its k most similar records per source,
            catching owners that a generic-looking S2/S3 record ranked lower.
Measured on train (US): 97.9% of true pairs survive, ~20 candidates per S1.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

HV = HashingVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), n_features=2 ** 24,
                       alternate_sign=False, norm=None, dtype=np.float32)


def _hash(texts):
    return HV.transform(texts)


def hashed_counts(texts, pool):
    chunks = [texts[s:s + 200_000] for s in range(0, len(texts), 200_000)]
    return sp.vstack(pool.map(_hash, chunks)).tocsr()


def tfidf(X):
    return TfidfTransformer(sublinear_tf=True).fit_transform(X).astype(np.float32).tocsr()


def prune_common(X, max_df_frac):
    """Drop very common columns from the *query* side. They barely move a
    cosine score but dominate the cost of the sparse product."""
    df = np.bincount(X.indices, minlength=X.shape[1])
    Y = (X @ sp.diags((df <= max_df_frac * X.shape[0]).astype(np.float32))).tocsr()
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


def block_country(d, k_rev, k_fwd, prune, pool):
    """d: normalised records of ONE country (all sources). Returns candidate
    pairs (q = S1 position, i = S2/S3 position in d) with TF-IDF features."""
    t = time.time()
    names = (d.core + " " + d.alt).str.strip().tolist()
    addrs = [" ".join("@" + w for w in s.split()) for s in d.addr.tolist()]   # "@": address terms never collide with name terms
    Xn, Xa = hashed_counts(names, pool), hashed_counts(addrs, pool)
    W, Nw, Aw = tfidf(Xn + Xa), tfidf(Xn), tfidf(Xa)
    del Xn, Xa, names, addrs
    src = d.src.to_numpy()
    s1, pl = np.flatnonzero(src == 1), np.flatnonzero(src != 1)
    if not len(s1) or not len(pl):
        return None

    q, i, r = topk(prune_common(W[pl], prune), W[s1], k_rev)
    parts = [pd.DataFrame({"q": s1[i], "i": pl[q], "rank_rev": r})]
    Wq = prune_common(W[s1], prune)
    for s in (2, 3):
        ps = np.flatnonzero(src == s)
        if len(ps):
            q, i, r = topk(Wq, W[ps], k_fwd)
            parts.append(pd.DataFrame({"q": s1[q], "i": ps[i], "rank_fwd": r}))
    c = pd.concat(parts, ignore_index=True).groupby(["q", "i"], sort=False).min().reset_index()
    c["rank_rev"] = c.rank_rev.fillna(k_rev + 1)                   # k+1 = "not retrieved this way"
    c["rank_fwd"] = c.rank_fwd.fillna(k_fwd + 1)
    q, i = c.q.to_numpy(), c.i.to_numpy()
    c["src"] = src[i].astype(np.int8)
    c["cos_w"] = rowdot(W, q, i)
    c["cos_nw"] = rowdot(Nw, q, i)
    c["cos_aw"] = rowdot(Aw, q, i)
    c["unm_a"] = unmatched_mass(Nw, q, i)
    c["unm_b"] = unmatched_mass(Nw, i, q)
    add_context(c)
    print(f"    {len(s1):,} S1 x {len(pl):,} S2/S3 -> {len(c):,} pairs "
          f"({len(c) / len(s1):.1f}/S1) in {time.time() - t:.0f}s", flush=True)
    return c


def add_context(c):
    """Where does this pair stand among its competitors?"""
    for col in ("cos_w", "cos_nw", "cos_aw"):
        g = c.groupby(["q", "src"])[col]
        c[f"{col}_gap"] = g.transform("max") - c[col]              # distance to this S1's best candidate
        c[f"{col}_rk"] = g.rank(ascending=False, method="min").astype(np.float32)
        r = c.groupby("i")[col]
        c[f"{col}_rgap"] = r.transform("max") - c[col]             # distance to the candidate's best S1
    c["n_s1_for_i"] = c.groupby("i").q.transform("size").astype(np.float32)
    c["n_high"] = c.assign(h=c.cos_w > 0.5).groupby(["q", "src"]).h.transform("sum").astype(np.float32)
