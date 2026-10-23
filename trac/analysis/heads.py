"""Correctness heads and the leakage-free evaluation protocol (CPU).

Every learned method is scored out of fold: 5 shuffled stratified 5-fold partitions (seeds
SEEDS), scores averaged over partitions. Within a cell all methods share the same folds.

  oof          fixed logistic head (controls, cue robustness)
  nested_oof   L2 strength / tree depth chosen by 3-fold CV inside each outer training fold
  stacked_oof  the TrAC head: cross-fitted stacking of base learners (logistic + boosted
               trees with fixed a-priori settings) under a low-dimensional logistic combiner
               that may also take raw anchor features; outer test folds never influence any fit

Base-learner predictions are cached under EXP_ROOT/base_cache by content hash (inputs, labels,
learner, seed); the cache is bit-identical to recomputation.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trac.config import BASE_CACHE

SEEDS = [2026, 7, 13, 42, 100]

# a-priori base-learner settings of the stacked head
FIXED = {"lr": {"logisticregression__C": 0.1}, "lr_hs": {"logisticregression__C": 0.01},
         "gb": {"max_depth": 3, "min_samples_leaf": 20}}


def clean(x):
    """Median-impute non-finite entries and append missingness indicators."""
    x = np.asarray(x, float).reshape(len(x), -1).copy()
    miss = ~np.isfinite(x)
    if miss.any():
        col = np.nanmedian(np.where(miss, np.nan, x), axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        x[miss] = np.take(col, np.where(miss)[1])
        x = np.hstack([x, miss[:, miss.any(0)].astype(float)])   # explicit missingness indicators
    return x


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def n_folds(y):
    return min(5, int(min(y.sum(), (1 - y).sum())))


def oof(X, y, C=1.0):
    """Out-of-fold scores of a standardized logistic head with fixed C."""
    X = clean(X)
    nf = n_folds(y)
    if nf < 2:
        return None
    a = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(nf, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        a += o
    return a / len(SEEDS)


def model(kind):
    """(estimator, tuning grid) of a head kind: lr, platt, lr_hs, or gb."""
    if kind == "lr":
        return (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                {"logisticregression__C": [0.01, 0.1, 1.0]})
    if kind == "platt":                       # one-feature recalibration: fixed, weak L2
        return (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                {"logisticregression__C": [100.0]})
    if kind == "lr_hs":                       # high-dimensional block: strong L2 range
        return (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                {"logisticregression__C": [0.001, 0.01, 0.1]})
    return (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                           early_stopping=False, random_state=0),
            {"max_depth": [2, 3], "min_samples_leaf": [20, 40]})


def nested_oof(X, y, kind, seeds=SEEDS):
    X = clean(X)
    nf = n_folds(y)
    if nf < 3:
        return None
    acc = np.zeros(len(y))
    for s in seeds:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(nf, shuffle=True, random_state=s).split(X, y):
            pipe, grid = model(kind)
            gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=s + 1),
                              scoring="roc_auc", n_jobs=1).fit(X[tr], y[tr])
            o[te] = gs.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(seeds)


def fit_fixed(kind, X, y):
    pipe, _ = model(kind)
    return pipe.set_params(**FIXED[kind]).fit(X, y)


def base_preds(X, kind, y, seed, nf):
    """Inner 3-fold OOF scores and refit test scores of one base learner for every outer
    fold of `seed` (deterministic given X, kind, y, seed), cached on disk by content hash."""
    h = hashlib.sha1(np.ascontiguousarray(X).tobytes() + y.tobytes() + f"{kind}|{seed}|{nf}|{FIXED[kind]}".encode())
    path = os.path.join(BASE_CACHE, h.hexdigest() + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        return [(z[f"i{f}"], z[f"t{f}"]) for f in range(nf)]
    out = []
    for tr, te in StratifiedKFold(nf, shuffle=True, random_state=seed).split(X, y):
        inner = np.zeros(len(tr))
        for itr, ite in StratifiedKFold(3, shuffle=True, random_state=seed + 1).split(X[tr], y[tr]):
            inner[ite] = fit_fixed(kind, X[tr][itr], y[tr][itr]).predict_proba(X[tr][ite])[:, 1]
        out.append((inner, fit_fixed(kind, X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    os.makedirs(BASE_CACHE, exist_ok=True)
    np.savez(path + ".tmp.npz", **{f"i{f}": a for f, (a, _) in enumerate(out)},
             **{f"t{f}": b for f, (_, b) in enumerate(out)})
    os.replace(path + ".tmp.npz", path)
    return out


def combiner():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def stacked_folds(bases, anchors, y, seed, nf):
    """Per outer fold of `seed`: (test indices, combiner scores on the test fold)."""
    bp = [base_preds(X, k, y, seed, nf) for X, k in bases]
    for f, (tr, te) in enumerate(StratifiedKFold(nf, shuffle=True, random_state=seed).split(anchors, y)):
        inner = np.column_stack([b[f][0] for b in bp]) if bases else np.zeros((len(tr), 0))
        test = np.column_stack([b[f][1] for b in bp]) if bases else np.zeros((len(te), 0))
        comb = combiner().fit(np.hstack([logit(inner), anchors[tr]]), y[tr])
        yield te, comb.predict_proba(np.hstack([logit(test), anchors[te]]))[:, 1]


def prepare_stack(bases, anchors, n):
    """Cleaned base inputs and anchors of a stacked head."""
    return [(clean(X), k) for X, k in bases], (clean(anchors) if anchors is not None else np.zeros((n, 0)))


def stacked_oof(bases, anchors, y, seeds=SEEDS):
    """Cross-fitted stacking. Inside each outer training fold, every base learner (X, kind)
    produces 3-fold inner out-of-fold scores; a low-dimensional logistic combiner is fit on
    [logit(base scores), anchors]. At test time the bases are refit on the whole outer
    training fold. Outer test folds never influence any fit."""
    bases, A = prepare_stack(bases, anchors, len(y))
    nf = n_folds(y)
    if nf < 3:
        return None
    acc = np.zeros(len(y))
    for s in seeds:
        o = np.zeros(len(y))
        for te, p in stacked_folds(bases, A, y, s, nf):
            o[te] = p
        acc += o
    return acc / len(seeds)


def stack_fit_predict(Xtr, ytr, Xte):
    """Fit the stacked LR + GB head on (Xtr, ytr) and score Xte (transfer experiments)."""
    X = clean(np.vstack([Xtr, Xte])); Xtr, Xte = X[:len(Xtr)], X[len(Xtr):]
    inner = np.zeros((len(ytr), 2))
    for itr, ite in StratifiedKFold(3, shuffle=True, random_state=1).split(Xtr, ytr):
        for j, k in enumerate(["lr", "gb"]):
            inner[ite, j] = fit_fixed(k, Xtr[itr], ytr[itr]).predict_proba(Xtr[ite])[:, 1]
    test = np.column_stack([fit_fixed(k, Xtr, ytr).predict_proba(Xte)[:, 1] for k in ["lr", "gb"]])
    comb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(logit(inner), ytr)
    return comb.predict_proba(logit(test))[:, 1]
