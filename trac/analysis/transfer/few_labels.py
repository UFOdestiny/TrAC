"""Label-free transfer against a small labeled target set (CPU).

For every eligible setting with at least 500 responses and each of five draws, 200 random target
responses are held out for testing, n of the remaining responses are labeled (n = 50, 100, 250, all),
and the test responses are scored by
  in-domain (n)    TrAC trained on the n target labels only (stacked classifier, and logistic regression)
  label-free       the LOMDO classifier with rank features (no target label; EXP_ROOT/transfer/lomdo_rank)
  pooled (n)       the LOMDO source settings plus the n target labels, rank features

  python -m trac.analysis.transfer.few_labels --tag gsm8k_qwen8b_k8   -> EXP_ROOT/transfer/few_labels/{tag}.json
  python -m trac.analysis.transfer.few_labels --aggregate             -> EXP_ROOT/reports/few_labels.json
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.heads import fit_fixed, stack_fit_predict
from trac.analysis.methods import trac_f
from trac.analysis.transfer.common import load_all, rank_norm, sources
from trac.benchmarks import parse_tag

OUT = C.TRANSFER / "few_labels"
NS, DRAWS, MIN_N, N_TEST = (50, 100, 250, "all"), 5, 500, 200


def run_tag(tag):
    d, m = parse_tag(tag)
    cells = load_all()
    if (d, m) not in cells or len(cells[(d, m)]["y"]) < MIN_N:
        print(f"[few] {tag} skipped")
        return
    y = cells[(d, m)]["y"].astype(int)
    Xr = rank_norm(trac_f(cells[(d, m)]))
    Xraw = np.asarray(trac_f(cells[(d, m)]), float)
    lf = np.load(C.TRANSFER / "lomdo_rank" / f"{tag}.npz")["TrAC"]
    src = sources(cells, d, m, "lomdo")
    Xs = np.vstack([rank_norm(trac_f(cells[k])) for k in src])
    ys = np.concatenate([cells[k]["y"] for k in src]).astype(int)
    lf_lr = fit_fixed("lr", Xs, ys).predict_proba(Xr)[:, 1]
    rng = np.random.RandomState(0)
    res = {str(n): {k: [] for k in ("in_lr", "in", "lf", "lf_lr", "pool")} for n in NS}
    for _ in range(DRAWS):
        while True:
            idx = rng.permutation(len(y))
            te, rest = idx[:N_TEST], idx[N_TEST:]
            if min(y[rest[: NS[0]]].sum(), (1 - y[rest[: NS[0]]]).sum()) >= 3:
                break
        for n in NS:
            tr = rest if n == "all" else rest[:n]
            r = res[str(n)]
            r["in_lr"].append(roc_auc_score(y[te], fit_fixed("lr", Xraw[tr], y[tr]).predict_proba(Xraw[te])[:, 1]))
            r["in"].append(
                roc_auc_score(y[te], stack_fit_predict(Xraw[tr], y[tr], Xraw[te])) if n == "all" or n >= 100 else np.nan
            )
            r["lf"].append(roc_auc_score(y[te], lf[te]))
            r["lf_lr"].append(roc_auc_score(y[te], lf_lr[te]))
            r["pool"].append(
                roc_auc_score(y[te], stack_fit_predict(np.vstack([Xs, Xr[tr]]), np.concatenate([ys, y[tr]]), Xr[te]))
                if n != "all" and n <= 100
                else np.nan
            )
    OUT.mkdir(parents=True, exist_ok=True)
    out = {n: {k: float(np.mean(v)) if np.isfinite(v).all() else None for k, v in r.items()} for n, r in res.items()}
    json.dump(out, open(OUT / f"{tag}.json", "w"), indent=1)
    print(tag, out, flush=True)


def aggregate():
    R = [json.load(open(f)) for f in sorted(glob.glob(str(OUT / "*.json")))]
    agg = {}
    for n in [str(x) for x in NS]:
        rs = [r[n] for r in R if n in r]
        agg[n] = {
            k: float(np.nanmean([x[k] for x in rs if x[k] is not None])) if any(x[k] is not None for x in rs) else None
            for k in rs[0]
        }
        agg[n]["n_settings"] = len(rs)
    agg["n_settings"] = len(R)
    print(json.dumps(agg, indent=1))
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump(agg, open(C.REPORTS / "few_labels.json", "w"), indent=1)


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
