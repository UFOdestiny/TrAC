"""Within-fold AUROC: robustness check for the pooled out-of-fold evaluation (CPU).

Pooling out-of-fold scores across folds compares scores from different fitted models; for
heavily tied features (e.g. saturated vote shares) the per-fold intercepts break ties in a way
that is anti-correlated with labels under stratified CV, which lowers the pooled AUROC of
learned methods. Here AUROC is computed inside every outer test fold (all scores from one
model) and averaged over folds and the five partition seeds; raw scores use the same folds.

  python -m trac.analysis.within_fold --tag T     -> EXP_ROOT/within_fold/T.json
  python -m trac.analysis.within_fold --aggregate
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from trac import config as C
from trac.analysis.cells import load_cell, load_hidden
from trac.analysis.heads import SEEDS, clean, model, n_folds, prepare_stack, stacked_folds
from trac.analysis.methods import METHODS, raw_baselines

LEARNED = ["TrAC", "TUP", "PCE", "Supervised prior signals", "TrAC + HS", "TUP + HS", "HS-probe",
           "Liu et al. (2024)", "TrAC + CF@8", "TUP + CF@8", "Supervised Consensus@8"]
RAW = ["TrAC-0", "SelfDoubt", "P(True)", "SC@8 (primary)", "CISC@8", "Semantic entropy@8 (lik.)"]


def _auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def fold_aucs_stack(X, y, s, nf):
    bases, A = prepare_stack(X["bases"], X["anchors"], len(y))
    return [_auc(y[te], p) for te, p in stacked_folds(bases, A, y, s, nf)]


def fold_aucs_nested(X, y, kind, s, nf):
    X = clean(X); out = []
    for tr, te in StratifiedKFold(nf, shuffle=True, random_state=s).split(X, y):
        pipe, grid = model(kind)
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=s + 1),
                          scoring="roc_auc", n_jobs=1).fit(X[tr], y[tr])
        out.append(_auc(y[te], gs.predict_proba(X[te])[:, 1]))
    return out


def run_tag(tag):
    B = load_cell(tag)
    y = B["y"].astype(int)
    nf = n_folds(y)
    if nf < 3:
        return
    H = load_hidden(tag, list(B["ids"]))
    raw = {}
    for d in raw_baselines(B).values():
        raw.update(d)
    res = {}
    for name in LEARNED + RAW:
        vals = []
        for s in SEEDS:
            if name in RAW:
                r = raw.get(name)
                if r is None or not np.isfinite(r).all():
                    break
                vals += [_auc(y[te], r[te]) for _, te in StratifiedKFold(nf, shuffle=True, random_state=s).split(r, y)]
                continue
            m = METHODS[name]
            X = m.build(B, H)
            if X is None:
                break
            vals += fold_aucs_stack(X, y, s, nf) if m.head == "stack" else fold_aucs_nested(X, y, m.head, s, nf)
        if vals:
            res[name] = float(np.nanmean(vals))
    C.WITHIN_FOLD.mkdir(parents=True, exist_ok=True)
    json.dump({"y_err": int((1 - y).sum()), "auc": res}, open(C.WITHIN_FOLD / f"{tag}.json", "w"))
    print("[fold]", tag, flush=True)


def aggregate(min_err=10):
    R = {}
    for f in sorted(glob.glob(str(C.WITHIN_FOLD / "*.json"))):
        j = json.load(open(f))
        if j["y_err"] < min_err:
            continue
        R[os.path.basename(f)[:-5]] = j["auc"]
    print(f"{len(R)} settings (within-fold AUROC, macro)")
    for n in LEARNED + RAW:
        v = [r[n] for r in R.values() if n in r]
        if v:
            print(f"{n:36s} {np.mean(v):.3f}  (n={len(v)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.tag:
        run_tag(a.tag)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
