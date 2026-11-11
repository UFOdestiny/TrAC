"""Hidden-state probe transferred through a label-free alignment of representation spaces (CPU).

Every backbone answers the same questions, so for a target (dataset d, backbone m) and each source
backbone m', canonical correlation analysis is fitted on the answer-end hidden states (layer 0.75L,
reduced to 256 principal components per backbone) of the two backbones' responses to the same
questions of the other datasets, which needs no label. Source responses on the other datasets are
mapped to the canonical space with the source side, the target's responses on d with the target side,
each canonical variate is ranked within its setting, and a logistic probe fitted on the sources
(LOMDO) scores the target.

  python -m trac.analysis.transfer.probe --all 6       -> EXP_ROOT/transfer/lomdo_probe/{tag}.npz
  python -m trac.analysis.transfer.probe --aggregate   -> EXP_ROOT/reports/probe_transfer.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from trac import config as C
from trac.analysis.cells import load_cell, load_hidden
from trac.analysis.transfer.common import load_all, rank_norm
from trac.benchmarks import DATASET_KEYS, MODEL_KEYS, parse_tag, tag_of
from trac.store import save_npz_atomic

OUT = C.TRANSFER / "lomdo_probe"
K, R = 32, 256


def backbone_states(m, cells_all):
    """{dataset: (ids, states)} of backbone m over every dataset (eligible or not), PCA-reduced jointly."""
    per = {}
    for d in DATASET_KEYS:
        B = cells_all.get((d, m)) or load_cell(tag_of(d, m))
        H = load_hidden(tag_of(d, m), list(B["ids"]))
        per[d] = ([str(q) for q in B["ids"]], H[:, 1, 1].astype(np.float32), B["y"].astype(int))
    X = np.vstack([v[1] for v in per.values()])
    sc = StandardScaler().fit(X)
    pca = PCA(R, random_state=0).fit(sc.transform(X))
    return {d: (ids, pca.transform(sc.transform(Hs)), y) for d, (ids, Hs, y) in per.items()}


_S, _CELLS = None, None


def run_tag(tag):
    d, m = parse_tag(tag)
    cells = _CELLS or load_all()
    if (d, m) not in cells:
        print(f"[probe] {tag} not eligible")
        return
    S = _S if _S is not None else {b: backbone_states(b, cells) for b in MODEL_KEYS}
    Xtr, ytr, target = [], [], []
    for b in [x for x in MODEL_KEYS if x != m]:
        A_, B_ = [], []
        for dd in DATASET_KEYS:
            if dd == d:
                continue
            ia, ib = S[b][dd][0], S[m][dd][0]
            pos = {q: i for i, q in enumerate(ib)}
            pairs = [(i, pos[q]) for i, q in enumerate(ia) if q in pos]
            A_.append(S[b][dd][1][[p[0] for p in pairs]])
            B_.append(S[m][dd][1][[p[1] for p in pairs]])
        cca = CCA(K, max_iter=1000).fit(np.vstack(A_), np.vstack(B_))
        for dd in DATASET_KEYS:
            if dd == d or (dd, b) not in cells:
                continue
            u = cca.transform(S[b][dd][1])
            Xtr.append(rank_norm(u))
            ytr.append(S[b][dd][2])
        _, v = cca.transform(S[b][d][1], S[m][d][1])
        target.append(rank_norm(v))
    y = S[m][d][2]
    Xt = np.mean(target, axis=0)  # target variates averaged over the three alignments
    clf = LogisticRegression(C=0.1, max_iter=3000).fit(np.vstack(Xtr), np.concatenate(ytr))
    p = clf.predict_proba(Xt)[:, 1]
    OUT.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(OUT / f"{tag}.npz", y=y, probe=p)
    print(f"[probe] {tag} aligned probe={roc_auc_score(y, p):.3f}", flush=True)


def aggregate():
    R_ = {}
    for f in sorted(glob.glob(str(OUT / "*.npz"))):
        z = np.load(f)
        R_[os.path.basename(f)[:-4]] = float(roc_auc_score(z["y"], z["probe"]))
    print(len(R_), float(np.mean(list(R_.values()))))
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump({"per": R_, "macro": float(np.mean(list(R_.values())))}, open(C.REPORTS / "probe_transfer.json", "w"), indent=1)


def run_all(jobs):
    global _S, _CELLS
    from multiprocessing import Pool

    _CELLS = load_all()
    _S = {b: backbone_states(b, _CELLS) for b in MODEL_KEYS}
    with Pool(jobs) as P:  # forked workers share the loaded states
        P.map(run_tag, [tag_of(d, m) for (d, m) in _CELLS], chunksize=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--all", type=int, default=0, help="every target with this many processes")
    a = ap.parse_args()
    if a.all:
        run_all(a.all)
    elif a.aggregate:
        aggregate()
    elif a.tag:
        run_tag(a.tag)
    else:
        ap.error("pass --tag, --all, or --aggregate")


if __name__ == "__main__":
    main()
