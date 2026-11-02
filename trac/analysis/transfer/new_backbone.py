"""A larger new backbone (CPU): Qwen3-32B scored by classifiers fitted on the four backbones of the main
matrix only (none of its labels), under LOMDO (the source settings of the other datasets) with rank
features, against training-free scores, the transferred views, and the full-trace verifier, with its
in-domain classifiers for reference.

  python -m trac.analysis.transfer.new_backbone      -> EXP_ROOT/reports/new_backbone_qwen32b.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.cells import load_cell
from trac.analysis.heads import SEEDS, stack_fit_predict, stacked_oof
from trac.analysis.methods import raw_baselines
from trac.analysis.stats import hboot
from trac.analysis.transfer.common import FREE, VIEWS, eligible, load_all, rank_norm, sources
from trac.analysis.transfer.verifier import verifier_scores
from trac.benchmarks import DATASET_KEYS, NEW_BACKBONE, tag_of

TOKEN, EARLY = "Token probs.", "Token probs. + early-exit signals"
TRANSFER_VIEWS = [TOKEN, EARLY, "TrAC", "TrAC (minimal)"]
IN_DOMAIN_VIEWS = [TOKEN, EARLY, "TrAC"]
PAIRS = [
    ("TrAC (LOMDO)", f"{TOKEN} (LOMDO)"),
    ("TrAC (LOMDO)", f"{EARLY} (LOMDO)"),
    ("TrAC (LOMDO)", "Output length"),
    ("TrAC (LOMDO)", "Reward model"),
    ("TrAC (LOMDO)", "Verifier, full trace"),
    ("TrAC + verifier", "TrAC (LOMDO)"),
    ("TrAC + verifier", "Verifier, full trace"),
    ("TrAC (in-domain)", f"{TOKEN} (in-domain)"),
    ("TrAC (in-domain)", f"{EARLY} (in-domain)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=NEW_BACKBONE)
    ap.add_argument("--k", type=int, default=1, help="traces per question of its generation cache")
    a = ap.parse_args()
    cells = load_all()
    per, raw = {}, {}
    for d in DATASET_KEYS:
        tag = tag_of(d, a.model, a.k)
        if not (C.FEATURES / f"{tag}.npz").exists():
            continue
        B = load_cell(tag)
        if not eligible(B) or "CON" not in B or "TRAJ_4" not in B:
            print("[new backbone] skip", tag)
            continue
        y = B["y"].astype(int)
        src = sources(cells, d, a.model, "lomdo")
        out = {}
        for v in TRANSFER_VIEWS:
            f = VIEWS[v]
            Xtr = np.vstack([rank_norm(f(cells[k])) for k in src])
            ytr = np.concatenate([cells[k]["y"] for k in src]).astype(int)
            out[f"{v} (LOMDO)"] = stack_fit_predict(Xtr, ytr, rank_norm(f(B)))
        for v in IN_DOMAIN_VIEWS:
            X = VIEWS[v](B)
            out[f"{v} (in-domain)"] = stacked_oof([(X, "lr"), (X, "gb")], None, y, SEEDS)
        out.update({n: s for c in raw_baselines(B).values() for n, s in c.items() if n in FREE and np.isfinite(s).all()})
        full = verifier_scores("full", tag, B["ids"])
        if full is not None:
            out["Verifier, full trace"] = full
            out["TrAC + verifier"] = rankdata(out["TrAC (LOMDO)"]) + rankdata(full)
        raw[d] = (y, out)
        per[d] = {n: float(roc_auc_score(y, s)) for n, s in out.items()}
        print("[new backbone]", tag, len(y), round(float(y.mean()), 3), {n: round(x, 3) for n, x in per[d].items()}, flush=True)
    names = sorted({n for r in per.values() for n in r if all(n in x for x in per.values())})
    res = {"n": len(per), "per": per, "macro": {n: float(np.mean([r[n] for r in per.values()])) for n in names}, "pairs": {}}
    for x, b in PAIRS:
        if x in names and b in names:
            mean, lo, hi = hboot([(y, o[x], o[b]) for y, o in raw.values()], n=1000)
            res["pairs"][f"{x} - {b}"] = {"diff": mean, "lo": lo, "hi": hi, "wins": int(sum(r[x] > r[b] for r in per.values()))}
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(C.REPORTS / f"new_backbone_{a.model}.json", "w"), indent=1)
    print(
        json.dumps({"n": res["n"], "macro": {k: round(v, 3) for k, v in res["macro"].items()}, "pairs": res["pairs"]}, indent=1)
    )


if __name__ == "__main__":
    main()
