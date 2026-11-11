"""Text-only correctness classifier transferred across models (CPU), in the spirit of generalized
correctness models: an l2-regularized logistic regression on the external judge's response embedding
(the last-token hidden state of Skywork-Reward-V2, trac.pipeline.reward), fitted on the LOMDO
source settings (other backbones, other datasets) and applied to the target without target labels.
It needs no access to the generating model. Also reports the judge embedding stacked with TrAC.

  python -m trac.analysis.transfer.judge --tag gsm8k_qwen8b_k8   -> EXP_ROOT/transfer/lomdo_judge/{tag}.npz
  python -m trac.analysis.transfer.judge --aggregate              -> EXP_ROOT/reports/judge_transfer.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from trac import config as C
from trac.analysis.heads import stack_fit_predict
from trac.analysis.methods import trac_f
from trac.analysis.transfer.common import load_all, rank_norm, sources
from trac.benchmarks import parse_tag, tag_of
from trac.store import save_npz_atomic

OUT = C.TRANSFER / "lomdo_judge"


def emb(tag, ids):
    z = np.load(C.REWARD / f"{tag}.npz")
    pos = {str(q): i for i, q in enumerate(z["ids"])}
    return z["emb"][[pos[str(q)] for q in ids]].astype(np.float32)


def run_tag(tag):
    d, m = parse_tag(tag)
    cells = load_all()
    if (d, m) not in cells:
        print(f"[judge] {tag} not eligible")
        return
    src = sources(cells, d, m, "lomdo")
    E = {k: emb(tag_of(*k), cells[k]["ids"]) for k in src + [(d, m)]}
    Xs = np.vstack([E[k] for k in src])
    ys = np.concatenate([cells[k]["y"] for k in src]).astype(int)
    sc = StandardScaler().fit(Xs)
    clf = LogisticRegression(C=0.001, max_iter=3000).fit(sc.transform(Xs), ys)
    y = cells[(d, m)]["y"].astype(int)
    judge = clf.predict_proba(sc.transform(E[(d, m)]))[:, 1]
    # TrAC with the transferred judge score as one more (rank) feature; source judge scores out of fold
    js = np.zeros(len(ys))
    grp = np.concatenate([np.full(len(cells[k]["y"]), i) for i, k in enumerate(src)])
    for tr, te in GroupKFold(5).split(Xs, ys, groups=grp):
        s2 = StandardScaler().fit(Xs[tr])
        js[te] = (
            LogisticRegression(C=0.001, max_iter=3000).fit(s2.transform(Xs[tr]), ys[tr]).predict_proba(s2.transform(Xs[te]))[:, 1]
        )
    off = np.cumsum([0] + [len(cells[k]["y"]) for k in src])
    Ts = np.vstack(
        [np.hstack([rank_norm(trac_f(cells[k])), rank_norm(js[off[i] : off[i + 1], None])]) for i, k in enumerate(src)]
    )
    Tt = np.hstack([rank_norm(trac_f(cells[(d, m)])), rank_norm(judge[:, None])])
    both = stack_fit_predict(Ts, ys, Tt)
    OUT.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(OUT / f"{tag}.npz", y=y, judge=judge, trac_judge=both)
    print(f"[judge] {tag} judge={roc_auc_score(y, judge):.3f} TrAC+judge={roc_auc_score(y, both):.3f}", flush=True)


def aggregate():
    R = {}
    for f in sorted(glob.glob(str(OUT / "*.npz"))):
        z = np.load(f)
        R[os.path.basename(f)[:-4]] = {k: float(roc_auc_score(z["y"], z[k])) for k in ("judge", "trac_judge")}
    macro = {k: float(np.mean([r[k] for r in R.values()])) for k in ("judge", "trac_judge")}
    print(len(R), macro)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump({"per": R, "macro": macro}, open(C.REPORTS / "judge_transfer.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    elif a.tag:
        run_tag(a.tag)
    else:
        ap.error("pass --tag or --aggregate")


if __name__ == "__main__":
    main()
