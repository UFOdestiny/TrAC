"""Label-free transfer of correctness classifiers across backbones and datasets (CPU).

For a target setting (dataset d, backbone m), the TrAC classifier on a feature view is fitted on source
settings (trac.analysis.transfer.common.sources) and applied to the target without any target label.
Features are used raw (`none`) or replaced, within every setting, by their empirical CDF over that
setting's responses (`rank`), which needs only the unlabeled responses of the target. A rank feature is
invariant to any strictly increasing per-backbone transform of the raw feature (different length
budgets, tokenizers, or entropy levels), which a classifier fitted on other backbones cannot correct.

  python -m trac.analysis.transfer.evaluate --tag gsm8k_qwen8b_k8   -> EXP_ROOT/transfer/{protocol}_{norm}/{tag}.npz
  python -m trac.analysis.transfer.evaluate --aggregate              -> EXP_ROOT/reports/transfer.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.heads import stack_fit_predict
from trac.analysis.methods import raw_baselines
from trac.analysis.transfer.common import FREE, VIEWS, load_all, rank_norm, sources
from trac.benchmarks import MODEL_KEYS, parse_tag
from trac.store import save_npz_atomic

TOKEN, EARLY, ANSWERS = "Token probs.", "Token probs. + early-exit signals", "Intermediate answers"
# (protocol, normalization) -> views reported in the paper
RUNS = {
    ("lomdo", "rank"): list(VIEWS),
    ("lomdo", "none"): [TOKEN, EARLY, ANSWERS, "TrAC"],
    ("lomfo", "rank"): [TOKEN, EARLY, "TrAC", "TrAC (minimal)"],
    ("lomo", "rank"): [TOKEN, EARLY, "TrAC"],
    ("lodo", "rank"): [TOKEN, "TrAC"],
}


def run_tag(tag):
    d, m = parse_tag(tag)
    cells = load_all()
    if (d, m) not in cells:
        print(f"[transfer] {tag} not eligible")
        return
    has_rm = all("RM" in B for B in cells.values())
    for (protocol, norm), views in RUNS.items():
        src = sources(cells, d, m, protocol)
        f = rank_norm if norm == "rank" else (lambda X: np.asarray(X, float))
        out = {"y": cells[(d, m)]["y"].astype(int)}
        for v in views:
            if "RM" in v and not has_rm:
                continue
            Xtr = np.vstack([f(VIEWS[v](cells[k])) for k in src])
            ytr = np.concatenate([cells[k]["y"] for k in src]).astype(int)
            out[v] = stack_fit_predict(Xtr, ytr, f(VIEWS[v](cells[(d, m)])))
        save_npz_atomic(C.TRANSFER / f"{protocol}_{norm}" / f"{tag}.npz", **out)
        print(
            f"[transfer] {tag} {protocol}/{norm} "
            + " ".join(f"{v}={roc_auc_score(out['y'], out[v]):.3f}" for v in out if v != "y"),
            flush=True,
        )


def aggregate():
    cells = load_all()
    free = {k: {n: v for c in raw_baselines(B).values() for n, v in c.items() if n in FREE} for k, B in cells.items()}
    res = {}
    for protocol, norm in RUNS:
        S = {}
        for f in sorted(glob.glob(str(C.TRANSFER / f"{protocol}_{norm}" / "*.npz"))):
            z = dict(np.load(f))
            z.update({n: v for n, v in free[parse_tag(os.path.basename(f)[:-4])].items() if np.isfinite(v).all()})
            S[parse_tag(os.path.basename(f)[:-4])] = z
        names = sorted({n for z in S.values() for n in z if n != "y"})
        au = {n: {f"{d}_{m}": float(roc_auc_score(z["y"], z[n])) for (d, m), z in S.items() if n in z} for n in names}
        res[f"{protocol}_{norm}"] = {
            "n": len(S),
            "auroc": au,
            "macro": {n: float(np.mean(list(a.values()))) for n, a in au.items()},
            "by_model": {
                n: {
                    m: float(np.mean([v for t, v in a.items() if t.endswith(m)]))
                    for m in MODEL_KEYS
                    if any(t.endswith(m) for t in a)
                }
                for n, a in au.items()
            },
        }
        print(f"{protocol}_{norm}", len(S), {n: round(v, 3) for n, v in res[f"{protocol}_{norm}"]["macro"].items()})
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(C.REPORTS / "transfer.json", "w"), indent=1)


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
