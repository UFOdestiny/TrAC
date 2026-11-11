"""Paired macro-AUROC differences with hierarchical-bootstrap intervals (CPU).

  in_domain    out-of-fold scores (EXP_ROOT/scores) and training-free scores of the eligible settings
  label_free   LOMDO transfer with rank features (EXP_ROOT/transfer/lomdo_rank), against the same views
               with raw features (lomdo_none) and the transferred judge classifier (lomdo_judge)

Each pair reports the mean difference, its 95% interval, and the settings where the first method wins.

  python -m trac.analysis.evaluation.pairs --jobs 6      -> EXP_ROOT/reports/pairs.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.cells import load_cell
from trac.analysis.methods import raw_baselines
from trac.analysis.stats import hboot
from trac.benchmarks import MIN_CLASS_COUNT

TOKEN = "Token probs."
PAIRS = {
    "in_domain": [
        ("TrAC", TOKEN),
        ("TrAC", f"{TOKEN} + early-exit signals"),
        ("TrAC", f"{TOKEN} + intermediate answers"),
        ("TrAC", "Supervised prior signals"),
        ("TrAC", "Output length"),
        ("TrAC", "Reward model"),
        ("TrAC", "Abl: without rival answers"),
        (f"{TOKEN} + rival answers", TOKEN),
        (f"{TOKEN} + rival answers", f"{TOKEN} + verbalized top-k"),
        ("TrAC", "Abl: rival at the end only"),
        ("TrAC", "Abl: end of trace only"),
        ("TrAC", "Abl: logistic head"),
        ("TrAC + probes", "Wb-S"),
        ("TrAC + probes", f"{TOKEN} + probes"),
        ("TrAC + RM", f"{TOKEN} + RM"),
        ("TrAC", "Math PRM"),
        ("TrAC + PRM", f"{TOKEN} + PRM"),
        ("TrAC", "TrAC (minimal)"),
        ("TrAC@8", "SC@8"),
        ("TrAC@8", f"{TOKEN} + consensus@8"),
    ],
    "label_free": [
        ("TrAC", TOKEN),
        ("TrAC", f"{TOKEN} + early-exit signals"),
        ("TrAC", "Output length"),
        ("TrAC", "Reward model"),
        ("TrAC", "TrAC (raw)"),
        ("TrAC", "Judge classifier"),
        ("TrAC + RM", f"{TOKEN} + RM"),
    ],
}

_S = {}


def in_domain():
    S = {}
    for f in sorted(glob.glob(str(C.SCORES / "*.npz"))):
        z = dict(np.load(f))
        if min(z["y"].sum(), (1 - z["y"]).sum()) < MIN_CLASS_COUNT:
            continue
        for c in raw_baselines(load_cell(os.path.basename(f)[:-4])).values():
            z.update({k: v for k, v in c.items() if np.isfinite(v).all()})
        S[os.path.basename(f)[:-4]] = z
    return S


def label_free():
    S = {}
    for f in sorted(glob.glob(str(C.TRANSFER / "lomdo_rank" / "*.npz"))):
        tag = os.path.basename(f)[:-4]
        z = dict(np.load(f))
        for c in raw_baselines(load_cell(tag)).values():
            z.update({k: v for k, v in c.items() if np.isfinite(v).all()})
        z["TrAC (raw)"] = np.load(C.TRANSFER / "lomdo_none" / f"{tag}.npz")["TrAC"]
        z["Judge classifier"] = np.load(C.TRANSFER / "lomdo_judge" / f"{tag}.npz")["judge"]
        S[tag] = z
    return S


def contrast(job):
    protocol, a, b = job
    S = _S[protocol]
    ks = [k for k, z in S.items() if a in z and b in z]
    mean, lo, hi = hboot([(S[k]["y"], S[k][a], S[k][b]) for k in ks], n=2000)
    wins = sum(roc_auc_score(S[k]["y"], S[k][a]) > roc_auc_score(S[k]["y"], S[k][b]) for k in ks)
    return protocol, f"{a} - {b}", {"diff": mean, "lo": lo, "hi": hi, "wins": int(wins), "n": len(ks)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    _S.update(in_domain=in_domain(), label_free=label_free())
    jobs = [(p, x, y) for p, pairs in PAIRS.items() for x, y in pairs]
    with Pool(a.jobs) as pool:  # forked workers share the loaded scores
        rows = pool.map(contrast, jobs, chunksize=1)
    res = {p: {} for p in PAIRS}
    for p, key, v in rows:
        res[p][key] = v
        print(f"{p:10s} {key:60s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] {v['wins']}/{v['n']}")
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(C.REPORTS / "pairs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
