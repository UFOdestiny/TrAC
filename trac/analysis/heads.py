"""Correctness classifiers and the leakage-free evaluation protocol (CPU).

Every learned method is scored out of fold: 5 shuffled stratified 5-fold partitions (seeds SEEDS),
scores averaged over partitions. Within a setting all methods share the same folds.

  nested_oof   L2 strength / tree depth chosen by 3-fold CV inside each outer training fold
  stacked_oof  the TrAC classifier: cross-fitted stacking of base learners (logistic regression and
               boosted trees with fixed a-priori settings) under a logistic meta-classifier that may
               also take raw anchor features; outer test folds never influence any fit
  stack_fit    the same classifier fitted once on source settings (label-free transfer)

Base-learner predictions are cached under EXP_ROOT/base_cache by content hash (inputs, labels,
learner, seed, preprocessing); the cache is bit-identical to recomputation.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trac.config import BASE_CACHE
from trac.store import save_npz_atomic

SEEDS = [2026, 7, 13, 42, 100]
PREPROCESSING = "fold-local-imputation-v1"  # part of the base-learner cache key

# a-priori base-learner settings of the stacked head
FIXED = {
    "lr": {"logisticregression__C": 0.1},
    "lr_hs": {"logisticregression__C": 0.01},
    "gb": {"histgradientboostingclassifier__max_depth": 3, "histgradientboostingclassifier__min_samples_leaf": 20},
}


def _matrix(x):
    x = np.asarray(x, float)
    return x.reshape(len(x), -1) if x.ndim != 2 else x


class FoldPreprocessor(TransformerMixin, BaseEstimator):
    """Learn imputation values and missing-indicator columns from fit rows only."""

    def fit(self, X, y=None):
        X = _matrix(X)
        self.n_features_in_ = X.shape[1]
        finite = np.isfinite(X)
        self.indicator_features_ = np.flatnonzero((~finite).any(axis=0))
        self.medians_ = np.array([np.median(col[ok]) if ok.any() else 0.0 for col, ok in zip(X.T, finite.T)])
        return self

    def transform(self, X):
        X = _matrix(X).copy()
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from the fitted preprocessing input")
        miss = ~np.isfinite(X)
        X[miss] = np.take(self.medians_, np.where(miss)[1])
        return np.hstack([X, miss[:, self.indicator_features_].astype(float)])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def n_folds(y):
    return min(5, int(min(y.sum(), (1 - y).sum())))


def model(kind):
    """(estimator, tuning grid) of a classifier kind: lr, lr_hs, or gb."""
    if kind == "lr":
        return (
            make_pipeline(FoldPreprocessor(), StandardScaler(), LogisticRegression(max_iter=3000)),
            {"logisticregression__C": [0.01, 0.1, 1.0]},
        )
    if kind == "lr_hs":  # high-dimensional block: strong L2 range
        return (
            make_pipeline(FoldPreprocessor(), StandardScaler(), LogisticRegression(max_iter=3000)),
            {"logisticregression__C": [0.001, 0.01, 0.1]},
        )
    return (
        make_pipeline(
            FoldPreprocessor(),
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, l2_regularization=1.0, early_stopping=False, random_state=0
            ),
        ),
        {"histgradientboostingclassifier__max_depth": [2, 3], "histgradientboostingclassifier__min_samples_leaf": [20, 40]},
    )


def nested_oof(X, y, kind, seeds=SEEDS):
    X = _matrix(X)
    nf = n_folds(y)
    if nf < 3:
        return None
    acc = np.zeros(len(y))
    for s in seeds:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(nf, shuffle=True, random_state=s).split(X, y):
            pipe, grid = model(kind)
            gs = GridSearchCV(
                pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=s + 1), scoring="roc_auc", n_jobs=1
            ).fit(X[tr], y[tr])
            o[te] = gs.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(seeds)


def fit_fixed(kind, X, y):
    pipe, _ = model(kind)
    return pipe.set_params(**FIXED[kind]).fit(X, y)


def base_preds(X, kind, y, seed, nf):
    """Inner 3-fold OOF scores and refit test scores of one base learner for every outer
    fold of `seed` (deterministic given X, kind, y, seed), cached on disk by content hash."""
    X = _matrix(X)
    h = hashlib.sha1(
        np.ascontiguousarray(X).tobytes() + y.tobytes() + f"{PREPROCESSING}|{X.shape}|{kind}|{seed}|{nf}|{FIXED[kind]}".encode()
    )
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
    save_npz_atomic(path, **{f"i{f}": a for f, (a, _) in enumerate(out)}, **{f"t{f}": b for f, (_, b) in enumerate(out)})
    return out


def combiner():
    return make_pipeline(FoldPreprocessor(), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def stacked_folds(bases, anchors, y, seed, nf):
    """Per outer fold of `seed`: (test indices, combiner scores on the test fold)."""
    bp = [base_preds(X, k, y, seed, nf) for X, k in bases]
    for f, (tr, te) in enumerate(StratifiedKFold(nf, shuffle=True, random_state=seed).split(anchors, y)):
        inner = np.column_stack([b[f][0] for b in bp]) if bases else np.zeros((len(tr), 0))
        test = np.column_stack([b[f][1] for b in bp]) if bases else np.zeros((len(te), 0))
        comb = combiner().fit(np.hstack([logit(inner), anchors[tr]]), y[tr])
        yield te, comb.predict_proba(np.hstack([logit(test), anchors[te]]))[:, 1]


def prepare_stack(bases, anchors, n):
    """Shape raw inputs without fitting preprocessing before outer or inner splits."""
    return [(_matrix(X), k) for X, k in bases], (_matrix(anchors) if anchors is not None else np.zeros((n, 0)))


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


def stack_fit(Xtr, ytr):
    """Fit the stacked LR + GB classifier on (Xtr, ytr) and return its scoring function (transfer)."""
    Xtr = _matrix(Xtr)
    inner = np.zeros((len(ytr), 2))
    for itr, ite in StratifiedKFold(3, shuffle=True, random_state=1).split(Xtr, ytr):
        for j, k in enumerate(["lr", "gb"]):
            inner[ite, j] = fit_fixed(k, Xtr[itr], ytr[itr]).predict_proba(Xtr[ite])[:, 1]
    bases = [fit_fixed(k, Xtr, ytr) for k in ["lr", "gb"]]
    comb = combiner().fit(logit(inner), ytr)
    return lambda X: comb.predict_proba(logit(np.column_stack([b.predict_proba(_matrix(X))[:, 1] for b in bases])))[:, 1]


def stack_fit_predict(Xtr, ytr, Xte):
    """Fit the stacked classifier on (Xtr, ytr) and score Xte."""
    return stack_fit(Xtr, ytr)(Xte)
