"""Out-of-fold scores of every learned method, and their macro evaluation (CPU).

  python -m trac.analysis.scores --tag T [--only "TrAC|TUP"] [--quick]
        score the learned methods of one cell -> EXP_ROOT/scores/{T}.npz
        (run cells in parallel processes; --only re-scores a subset and keeps the rest)
  python -m trac.analysis.scores --aggregate
        add the raw baselines and report macro AUROC / AURC / ECE / Brier / Risk@10% per
        budget class, with paired hierarchical-bootstrap contrasts of the best TrAC variant
        against the best baseline in each class -> EXP_ROOT/scores.json

Protocol (per cell): outer 5 x shuffled 5-fold CV with identical folds for every learned
method (trac.analysis.heads). Cells with fewer than --min-err correct or incorrect responses
are reported but excluded from aggregates.

Run single-threaded per process (OMP_NUM_THREADS=1); scripts/run_analysis.sh parallelizes
over cells.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.cells import load_cell, load_hidden
from trac.analysis.heads import SEEDS, nested_oof, stacked_oof
from trac.analysis.methods import METHODS, raw_baselines
from trac.analysis.stats import hboot
from trac.metrics import aurc, brier, ece, risk_at_coverage

CLASSES = ["trac0", "single-gray", "single-gray-pc", "single-white", "k2", "k8", "ablation"]


def score(name, B, H, y, seeds=SEEDS):
    """Out-of-fold scores of a learned method on one cell, or None if it does not apply."""
    m = METHODS[name]
    X = m.build(B, H)
    if X is None:
        return None
    if m.head == "stack":
        return stacked_oof(X["bases"], X["anchors"], y, seeds)
    return nested_oof(X, y, m.head, seeds)


def run_tag(tag, seeds, only=()):
    B = load_cell(tag)
    y = B["y"].astype(int)
    H = load_hidden(tag, list(B["ids"]))
    out = {"y": y}
    path = C.SCORES / f"{tag}.npz"
    if only and path.exists():               # incremental: keep previously scored methods
        out.update({k: v for k, v in np.load(path).items() if k != "y"})
    for name in METHODS:
        if only and name not in only:
            continue
        s = score(name, B, H, y, seeds)
        if s is not None:
            out[name] = s
        print(f"[scores] {tag} {name}", flush=True)
    C.SCORES.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def metrics(s, y):
    return {"auroc": roc_auc_score(y, s), "aurc": aurc(s, y), "ece": ece(s, y), "brier": brier(s, y),
            "risk@10": risk_at_coverage(s, y, 0.1)}


def aggregate(min_err):
    cells, Y, S, cls = [], {}, defaultdict(dict), {}
    for f in sorted(glob.glob(str(C.SCORES / "*.npz"))):
        t = os.path.basename(f)[:-4]
        o = dict(np.load(f))
        y = o.pop("y")
        if min(y.sum(), (1 - y).sum()) < min_err:
            continue
        cells.append(t); Y[t] = y
        for k, v in o.items():
            S[k][t] = v
            cls[k] = METHODS[k].budget
        for c, d in raw_baselines(load_cell(t)).items():
            for k, v in d.items():
                if np.isfinite(v).all():
                    S[k][t] = v; cls[k] = c
    res = {}
    print(f"{len(cells)} cells\n{'method':34s}{'class':>15s}{'AUROC':>7s}{'AURC':>7s}{'ECE':>7s}{'Brier':>7s}{'R@10':>7s}")
    for c in CLASSES:
        for k in [k for k in S if cls[k] == c]:
            m = {mk: float(np.mean([metrics(S[k][t], Y[t])[mk] for t in cells if t in S[k]]))
                 for mk in ["auroc", "aurc", "ece", "brier", "risk@10"]}
            raw = k not in METHODS
            res[k] = {**m, "class": c, "raw": raw, "ours": c == "trac0" or (not raw and METHODS[k].ours)}
            ece_s = "   -  " if raw else f"{m['ece']:7.3f}"
            bri_s = "   -  " if raw else f"{m['brier']:7.3f}"
            print(f"{k:34s}{c:>15s}{m['auroc']:7.3f}{m['aurc']:7.3f}{ece_s:>7s}{bri_s:>7s}{m['risk@10']:7.3f}")
    print("\nbest ours vs best baseline per class (AUROC, paired hierarchical bootstrap):")
    contrasts = {}
    for c in ["trac0", "single-gray", "single-white", "k8"]:
        ours = {k: v["auroc"] for k, v in res.items() if v["class"] == c and v["ours"]}
        base = {k: v["auroc"] for k, v in res.items() if v["class"] == c and not v["ours"]}
        if c == "trac0":          # training-free TrAC vs every training-free single-trace baseline
            ours = {k: v for k, v in ours.items() if k == "TrAC-0"}
            base = {k: v["auroc"] for k, v in res.items() if v["class"] == "single-gray" and v["raw"]}
        if c == "single-white":   # grey-box baselines are also valid competitors for white-box methods
            base.update({k: v["auroc"] for k, v in res.items() if v["class"] == "single-gray" and not v["ours"]})
        if not ours or not base:
            continue
        a, b = max(ours, key=ours.get), max(base, key=base.get)
        cl = [(Y[t], S[a][t], S[b][t]) for t in cells if t in S[a] and t in S[b]]
        m, lo, hi = hboot(cl)
        wins = sum(roc_auc_score(x[0], x[1]) > roc_auc_score(x[0], x[2]) for x in cl)
        contrasts[c] = {"ours": a, "baseline": b, "delta": m, "ci": [lo, hi], "wins": wins, "n": len(cl)}
        print(f"  {c:12s} {a} vs {b}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] wins {wins}/{len(cl)}")
    json.dump({"cells": cells, "methods": res, "contrasts": contrasts}, open(C.EXP_ROOT / "scores.json", "w"), indent=1)
    print("saved", C.EXP_ROOT / "scores.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", default="", help="|-separated method names to (re)score")
    ap.add_argument("--quick", action="store_true", help="one outer partition seed (screening)")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--min-err", type=int, default=10)
    args = ap.parse_args()
    if args.tag:
        run_tag(args.tag, SEEDS[:1] if args.quick else SEEDS, set(args.only.split("|")) - {""})
    if args.aggregate:
        aggregate(args.min_err)


if __name__ == "__main__":
    main()
