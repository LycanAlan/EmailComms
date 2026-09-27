"""Apply the validated decoy rule(s) to a finished matching_results.tsv.

  python apply_rules.py --matching IN.tsv --test-dir DATASET/test --out OUT.tsv [--fr-number]

Always applied : legal_swap (Inc<->Corp, LLC->Co). Validation (night model): removes pairs that are 95% false,
                 US macro F0.5 +0.00038; the night test file had 12x more such pairs per US S1 than validation.
--fr-number    : also drop France pairs whose S1 house number is not among the record's numbers. Without the flag
                 the script only REPORTS how many such pairs the file still has (the sub-10 file already applies
                 its own France house-number rule, so check the count before switching this on).
Every S1 row is kept (lists are only shortened), so the output stays valid for the portal.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rules

ap = argparse.ArgumentParser()
ap.add_argument("--matching", type=Path, required=True)
ap.add_argument("--test-dir", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--fr-number", action="store_true")
a = ap.parse_args()

rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
src = pd.concat([rd(a.test_dir / f"test_source{k}.tsv") for k in (1, 2, 3)]).set_index("entity_id")
res = rd(a.matching)
lists = res.matched_entity_ids.map(lambda x: x.split(",") if x else [])
q = np.repeat(res.source1_entity_id.to_numpy(), lists.str.len().to_numpy())
i = np.concatenate(lists.to_numpy()) if len(q) else np.array([], dtype=object)
pairs = pd.DataFrame({"q": q, "i": i})
N, A, C = src.business_name, src.business_address, src.country
pairs["cty"] = C.reindex(pairs.q).to_numpy()
qn, qa = N.reindex(pairs.q).to_numpy(), A.reindex(pairs.q).to_numpy()
rn, ra = N.reindex(pairs.i).to_numpy(), A.reindex(pairs.i).to_numpy()
pairs["legal_swap"] = [rules.legal_swap(x, y) for x, y in zip(qn, rn)]
pairs["fr_number"] = [c == "France" and rules.fr_number(x, y) for c, x, y in zip(pairs.cty, qa, ra)]
drop = pairs.legal_swap | (pairs.fr_number if a.fr_number else False)
n_s1 = C[C.index.str.startswith("S1-")].value_counts()
print(f"input: {len(pairs):,} matches")
for c in sorted(pairs.cty.dropna().unique()):
    m = pairs.cty == c
    print(f"  {c:7} matches {m.sum():>9,} | legal_swap {int((m & pairs.legal_swap).sum()):>7,} ({(m & pairs.legal_swap).sum() / n_s1[c]:.4f}/S1) | "
          f"fr_number {'REMOVED' if a.fr_number else 'would remove'} {int((m & pairs.fr_number).sum()):>7,}")
kept = pairs[~drop]
out = kept.groupby("q").i.agg(",".join).reindex(res.source1_entity_id).fillna("")
a.out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"source1_entity_id": res.source1_entity_id.to_numpy(), "matched_entity_ids": out.to_numpy()}).to_csv(a.out, sep="\t", index=False)
print(f"wrote {a.out}: {len(kept):,} matches ({int(drop.sum()):,} removed), {len(res):,} S1 rows")
