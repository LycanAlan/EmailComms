"""Generator fingerprints and name ambiguity: record-level signals that normalize.py erases.

The data generator applies its "sloppy entry" noise (dropped address, dropped house
number, lowercase, leetspeak, websites, aliases...) to TRUE copies far more often than
to decoys, and decoys are longer (an extra word) and keep their legal form. Measured on
train, S2/S3 records: an alias appears on 3.5% of true copies and 0.0% of decoys, a
website in the name 5.2% vs 0.6%, an empty address 4.5% vs 0.3%, an address without
numbers 12.4% vs 0.9%. Those patterns only exist in the RAW strings, so they are
computed here from the raw TSV columns, not from normalize.py's output.

Name ambiguity: how many S1 entities share a record's cleaned name (a candidate whose
name belongs to many S1 entities cannot be placed by name alone), and how much of a
name is made of words that appear in no S1 name at all (names replaced by made-up words
like "Onyxviocalo" carry no information, the address has to decide). Both are computed
per country from that split's own S1 records, without labels, so they work for France.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

RX = {  # RE2 patterns on the RAW strings, run in Arrow's C++ layer
    "alias": ("name", r"(?i)\b(formerly|aka|a/k/a|dba|d/b/a|nee|f/k/a|fka|trading as|t/a)\b"),
    "url": ("name", r"(?i)www\.|https?://|\.(com|net|org|in|fr|co)\b"),
    "ncaps": ("name", r"^[^a-z]*[A-Z][^a-z]*$"),
    "nlower": ("name", r"^[^A-Z]*[a-z][^A-Z]*$"),
    "leet": ("name", r"[A-Za-z][0-9][A-Za-z]|\b[0-9][A-Za-z]{3,}\b|\b[A-Za-z]{3,}[0-9]\b"),   # inner or edge digit: 5ervices, denta1
    "accent": ("name", r"[\x{00C0}-\x{024F}]"),
    "nlead": ("name", r"^\W"),
    "idtag": ("name", r"(?i)\bid\s*[:#]?\s*\d"),
    "phone": ("name", r"\+?\d[\d\s-]{8,}\d"),
    "ndbl": ("name", r"\s{2,}"),
    "aempty": ("addr", r"^\s*$"),
    "anonum": ("addr", r"^\D*$"),
    "azpad": ("addr", r"(^|\D)0\d"),
    "anull": ("addr", r"(?i)\bnull\b"),
    "alead": ("addr", r"^\W"),
}
FLAG_COLS = [f"fp_{k}" for k in RX] + ["fp_ntok", "fp_nlen", "fp_alen", "fp_acomma"]
AMB_COLS = ["amb_core", "amb_phon", "oov_share", "oov_n", "tw_sound", "tw_addr", "tw_name"]


def raw_flags(name, addr):
    """name, addr: pyarrow string arrays of RAW business_name / business_address -> one row per record."""
    col = {"name": name, "addr": addr}
    out = {f"fp_{k}": pc.match_substring_regex(col[c], rx).to_numpy(zero_copy_only=False).astype(np.int8)
           for k, (c, rx) in RX.items()}
    out["fp_ntok"] = pc.list_value_length(pc.utf8_split_whitespace(name)).to_numpy(zero_copy_only=False).astype(np.int16)
    out["fp_nlen"] = pc.utf8_length(name).to_numpy(zero_copy_only=False).astype(np.int16)
    out["fp_alen"] = pc.utf8_length(addr).to_numpy(zero_copy_only=False).astype(np.int16)
    out["fp_acomma"] = pc.count_substring(addr, ",").to_numpy(zero_copy_only=False).astype(np.int8)
    return pd.DataFrame(out)


def ambiguity(norm):
    """norm: normalised records (src, country, core, phon). Per record: number of S1 entities in the same
    country with the identical core name / phonetic key, and the share / count of name words that no S1
    name in that country uses."""
    n = len(norm)
    src = norm.src.to_numpy()
    cty = pa.array(norm.country.astype(str).to_numpy(), pa.large_string())
    out = {k: np.zeros(n, np.float32) for k in AMB_COLS}
    for col, name in (("core", "amb_core"), ("phon", "amb_phon")):
        a = pa.array(norm[col].astype(str).to_numpy(), pa.large_string())
        codes = pc.dictionary_encode(pc.binary_join_element_wise(cty, a, pa.scalar("|", pa.large_string()))).indices.to_numpy()
        cnt = np.bincount(codes[src == 1], minlength=codes.max() + 1)
        out[name] = cnt[codes].astype(np.float32)
        out[name][pc.equal(a, "").to_numpy(zero_copy_only=False)] = np.nan
    core = pa.array(norm.core.astype(str).to_numpy(), pa.large_string())
    c_np = norm.country.astype(str).to_numpy()
    for c in np.unique(c_np):
        rows = np.flatnonzero(c_np == c)
        toks = pc.utf8_split_whitespace(core.take(pa.array(rows)))
        vocab = pc.unique(pc.list_flatten(pc.utf8_split_whitespace(core.take(pa.array(rows[src[rows] == 1])))))
        oov = (~pc.is_in(pc.list_flatten(toks), value_set=vocab).to_numpy(zero_copy_only=False)).astype(np.float32)
        lens = pc.list_value_length(toks).to_numpy(zero_copy_only=False).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        s = np.add.reduceat(np.append(oov, 0), np.minimum(starts, len(oov))) * (lens > 0)
        out["oov_n"][rows] = s
        out["oov_share"][rows] = np.where(lens > 0, s / np.maximum(lens, 1), np.nan)
    # Twins: OTHER S2/S3 records of the country with the same name sound + numbers / address / name.
    # A true business has several copies that agree with each other, a decoy is a one-off: on US train
    # 44.9% of true copies have a sound+numbers twin, 1.2% of decoys.
    pl = src != 1
    txt = lambda c: pa.array(norm[c].astype(str).to_numpy(), pa.large_string())
    def twins(*cols):
        key = cty
        for c in cols:
            key = pc.binary_join_element_wise(key, txt(c), pa.scalar("|", pa.large_string()))
        codes = pc.dictionary_encode(key).indices.to_numpy()
        return (np.bincount(codes[pl], minlength=codes.max() + 1)[codes] - pl).astype(np.float32)
    has = lambda c: pc.greater(pc.utf8_length(txt(c)), 0).to_numpy(zero_copy_only=False)
    out["tw_sound"] = np.where(has("phon") & has("nums"), twins("phon", "nums"), np.nan)
    out["tw_addr"] = np.where(has("addr"), twins("addr"), np.nan)
    out["tw_name"] = np.where(has("core"), twins("core"), np.nan)
    return pd.DataFrame(out)


def pair_extras(f, norm, q, i):
    """Adds the fingerprint + ambiguity pair features to f (q = S1 row, i = candidate row in norm)."""
    col = lambda c: norm[c].to_numpy()
    for k in RX:
        f[f"fp_{k}"] = col(f"fp_{k}")[i]
    f["fp_ncaps_a"] = col("fp_ncaps")[q]
    f["fp_ntok_d"] = col("fp_ntok")[i].astype(np.float32) - col("fp_ntok")[q]          # decoys add a word
    f["fp_nlen_r"] = col("fp_nlen")[i] / np.maximum(col("fp_nlen")[q], 1).astype(np.float32)
    f["fp_alen_r"] = col("fp_alen")[i] / np.maximum(col("fp_alen")[q], 1).astype(np.float32)
    f["fp_acomma_d"] = col("fp_acomma")[i].astype(np.float32) - col("fp_acomma")[q]
    f["amb_core_q"] = col("amb_core")[q]
    f["amb_core_i"] = col("amb_core")[i]
    f["amb_phon_i"] = col("amb_phon")[i]
    f["oov_share_i"] = col("oov_share")[i]
    f["oov_n_i"] = col("oov_n")[i]
    for k in ("tw_sound", "tw_addr", "tw_name"):
        f[f"{k}_i"] = col(k)[i]
    f["tw_sound_q"] = col("tw_sound")[q]                                   # S2/S3 records carrying the S1's exact sound + numbers
    return f


def demo():
    df = raw_flags(pa.array(["Tevue Industries", "interfaithcenter.com", "Synarcumbra formerly known as X", "UNlTED FUND PC"]),
                   pa.array(["", "12 Main St", "Pune", "1694 33RD STREET, RIO RANCHO, NM"]))
    assert df.fp_aempty.tolist() == [1, 0, 0, 0] and df.fp_url.tolist() == [0, 1, 0, 0]
    assert df.fp_alias.tolist() == [0, 0, 1, 0] and df.fp_anonum.tolist() == [1, 0, 1, 0]
    assert df.fp_ncaps.tolist() == [0, 0, 0, 0] and df.fp_leet.tolist() == [0, 0, 0, 0]   # "UNlTED": lowercase l, not all caps
    norm = pd.DataFrame({"src": np.int8([1, 1, 2, 2]), "country": ["US"] * 4,
                         "core": ["hays furniture", "hays furniture", "hays furniture", "onyxviocalo"],
                         "phon": ["hs frntr", "hs frntr", "hs frntr", "onksvkl"],
                         "addr": ["12 main st", "40 oak ave", "12 main st", ""], "nums": ["12", "40", "12", ""]})
    A = ambiguity(norm)
    assert A.amb_core.tolist() == [2, 2, 2, 0] and A.oov_share.tolist() == [0, 0, 0, 1]
    assert A.tw_name.tolist() == [1, 1, 0, 0]                            # the one S2 named 'hays furniture'; itself excluded
    assert A.tw_sound.tolist()[:3] == [1, 0, 0] and np.isnan(A.tw_sound[3]) and np.isnan(A.tw_addr[3])
    print("fingerprints.demo OK")


if __name__ == "__main__":
    demo()
