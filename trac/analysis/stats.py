"""Paired hierarchical bootstrap over settings."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def hboot(cells, n=2000, seed=0):
    """Pair-equal hierarchical bootstrap of the macro AUROC difference.

    cells: list of (y, scores_a, scores_b). Resamples settings, then responses within each
    setting (stratified by label). Returns (mean difference, 2.5th, 97.5th percentile)."""
    rng = np.random.RandomState(seed); ds = []
    for _ in range(n):
        d = []
        for ci in rng.choice(len(cells), len(cells), True):
            y, a, b = cells[ci]
            ip, ng = np.where(y == 1)[0], np.where(y == 0)[0]
            ii = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ng, len(ng), True)])
            d.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
        ds.append(np.mean(d))
    ds = np.array(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
