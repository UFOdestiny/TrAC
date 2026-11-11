"""Label-free transfer to a new model: source settings and rank features.

A rank feature replaces every feature, in source and target settings alike, by its empirical
distribution function over that setting's responses, computed without labels. It is invariant to any
strictly increasing per-model transform of the raw feature (length budgets, tokenizers, entropy levels),
which a classifier fitted on other models cannot correct.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from rewind import config as C
from rewind.benchmarks import DATASET_KEYS, MATH_DATASETS, MIN_CLASS_COUNT, MODEL_KEYS, tag_of
from rewind.features.setting import load_setting


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
    """At least MIN_CLASS_COUNT correct and incorrect responses."""
    y = B["y"]
    return min(y.sum(), (1 - y).sum()) >= MIN_CLASS_COUNT


def load_settings():
    """Feature blocks of every eligible setting of the main matrix, keyed by (dataset, model)."""
    cells = {}
    for d in DATASET_KEYS:
        for m in MODEL_KEYS:
            if (C.FEATURES / f"{tag_of(d, m)}.npz").exists():
                B = load_setting(tag_of(d, m))
                if eligible(B):
                    cells[(d, m)] = B
    return cells


def sources(cells, d, m, protocol):
    """Source settings for the target (d, m): no shared model and dataset (lomdo), additionally no
    shared task family (lomfo), no shared model (lomo), or no shared dataset (lodo)."""
    fam = lambda x: x in MATH_DATASETS
    keep = {
        "lomdo": lambda k: k[0] != d and k[1] != m,
        "lomfo": lambda k: k[1] != m and fam(k[0]) != fam(d),
        "lomo": lambda k: k[1] != m,
        "lodo": lambda k: k[0] != d,
    }[protocol]
    return [k for k in cells if keep(k)]
