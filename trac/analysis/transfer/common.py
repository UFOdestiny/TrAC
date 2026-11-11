"""Shared pieces of label-free transfer: settings, source selection, feature views, and rank features."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from trac import config as C
from trac.analysis.cells import load_cell
from trac.analysis.methods import answer_f, early_exit_f, likelihood_f, minimal_f, token_f, trac_f
from trac.benchmarks import DATASET_KEYS, MATH_DATASETS, MIN_CLASS_COUNT, MODEL_KEYS, tag_of

VIEWS = {
    "Token probs.": token_f,
    "Token probs. + early-exit signals": lambda B: np.hstack([token_f(B), early_exit_f(B)]),
    "Token probs. + likelihood of a": lambda B: np.hstack([token_f(B), likelihood_f(B)]),
    "Intermediate answers": answer_f,
    "TrAC": trac_f,
    "TrAC (minimal)": minimal_f,
    "Token probs. + RM": lambda B: np.hstack([token_f(B), B["RM"]]),
    "TrAC + RM": lambda B: np.hstack([trac_f(B), B["RM"]]),
}
FREE = ["Output length", "Reward model", "Mean log-prob", "P(True)", "SelfDoubt"]  # training-free scores


def rank_norm(X):
    """Empirical CDF of every column within one setting (NaNs kept for imputation)."""
    X = np.asarray(X, float).copy()
    X[~np.isfinite(X)] = np.nan
    out = np.full_like(X, np.nan)
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j])
        if ok.sum() > 1:
            out[ok, j] = (rankdata(X[ok, j]) - 0.5) / ok.sum()
    return out


def eligible(B):
    y = B["y"]
    return min(y.sum(), (1 - y).sum()) >= MIN_CLASS_COUNT


def load_all():
    """Feature blocks of every eligible setting of the main matrix, keyed by (dataset, model)."""
    cells = {}
    for d in DATASET_KEYS:
        for m in MODEL_KEYS:
            if (C.FEATURES / f"{tag_of(d, m)}.npz").exists():
                B = load_cell(tag_of(d, m))
                if eligible(B):
                    cells[(d, m)] = B
    return cells


def sources(cells, d, m, protocol):
    """Source settings for the target (d, m): no shared backbone and dataset (lomdo), additionally no
    shared task family (lomfo), no shared backbone (lomo), or no shared dataset (lodo)."""
    fam = lambda x: x in MATH_DATASETS
    keep = {
        "lomdo": lambda k: k[0] != d and k[1] != m,
        "lomfo": lambda k: k[1] != m and fam(k[0]) != fam(d),
        "lomo": lambda k: k[1] != m,
        "lodo": lambda k: k[0] != d,
    }[protocol]
    return [k for k in cells if keep(k)]
