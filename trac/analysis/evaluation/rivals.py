"""What the rival answers reveal (CPU).

  reproduced   among responses whose greedy intermediate answer at the end of the trace reproduces a,
               where agreement carries no information, the AUROC of the rival share of a (CON)
  rival_gold   among incorrect responses, how often the rival elicited at the end of the trace, or any
               rival elicited along it, is the correct answer

Over the eligible settings of the main matrix, per backbone, and on the free-form benchmarks.

  python -m trac.analysis.evaluation.rivals --jobs 6      -> EXP_ROOT/reports/rivals.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import answers as A
from trac import config as C
from trac.analysis.cells import load_cell
from trac.benchmarks import DATASET_KEYS, MIN_CLASS_COUNT, MODEL_KEYS, parse_tag, tag_of
from trac.store import read_json_gz

END_AGREE = 20  # column of TRAJ: greedy agreement at the end of the trace
MULTIPLE_CHOICE = {"gpqa_diamond", "mmlupro"}


def setting(tag):
    B = load_cell(tag)
    y = B["y"].astype(int)
    if min(y.sum(), (1 - y).sum()) < MIN_CLASS_COUNT:
        return None
    G = {r["id"]: r for r in read_json_gz(C.GEN / f"{tag}.json.gz")}
    R = {r["id"]: r for r in read_json_gz(C.CONTRAST / f"{tag}.json.gz")}
    errors = end_gold = any_gold = 0
    for q, label in zip(B["ids"], y):
        g, r = G.get(str(q)), R.get(str(q))
        if label == 1 or g is None or r is None:
            continue
        errors += 1
        hits = [e["ans"] is not None and A.equiv(g, e["ans"], g.get("gold")) for e in r["elic"]]
        if hits:
            end_gold += hits[-1]
            any_gold += any(hits)
    share = B["CON"][:, 0]
    reproduced = (B["TRAJ"][:, END_AGREE] == 1) & np.isfinite(share)
    yr = y[reproduced]
    auroc = roc_auc_score(yr, share[reproduced]) if min(yr.sum(), (1 - yr).sum()) >= 5 else np.nan
    return tag, {
        "errors": errors,
        "end_gold": end_gold,
        "any_gold": any_gold,
        "auroc": auroc,
        "reproduced": float(reproduced.mean()),
    }


def summary(rows):
    errors = sum(r["errors"] for r in rows)
    return {
        "settings": len(rows),
        "reproduced_share_auroc": float(np.nanmean([r["auroc"] for r in rows])),
        "reproduced": float(np.mean([r["reproduced"] for r in rows])),
        "errors": errors,
        "end_rival_gold": sum(r["end_gold"] for r in rows) / errors,
        "any_rival_gold": sum(r["any_gold"] for r in rows) / errors,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    tags = [tag_of(d, m) for d in DATASET_KEYS for m in MODEL_KEYS if (C.CONTRAST / f"{tag_of(d, m)}.json.gz").exists()]
    with Pool(a.jobs) as pool:
        per = dict(r for r in pool.map(setting, tags, chunksize=1) if r)
    subsets = {"all": per, "free_form": {t: r for t, r in per.items() if parse_tag(t)[0] not in MULTIPLE_CHOICE}}
    res = {
        name: {
            "overall": summary(list(S.values())),
            "by_model": {m: summary([r for t, r in S.items() if parse_tag(t)[1] == m]) for m in MODEL_KEYS},
        }
        for name, S in subsets.items()
    }
    res["per"] = per
    print(json.dumps({k: v for k, v in res.items() if k != "per"}, indent=1))
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "rivals.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
