"""Diagnostics of label-free transfer (CPU), for the LOMDO protocol with rank features.

(1) Reference set. Rank features need unlabeled target responses to rank against. For each target
    setting, the classifier fitted on the source settings is applied with target ranks computed against
    (a) all responses of the setting, (b) random reference sets of n responses (5 draws), and
    (c) all responses of the target backbone across every dataset (no setting boundary),
    (d) the target backbone's responses to the other datasets only (no response of the target task), and
    (e) no setting boundary anywhere: source and target features ranked against their backbone's
        responses pooled over datasets, and the classifier refitted on these ranks.
(2) Ordinal shift. For every target and view, the in-domain out-of-fold score s_T serves as a proxy
    for the target's conditional correctness and the transferred score s_S as the source's; the
    proxy shift is min over increasing g of mean |s_T - g(s_S)| (isotonic regression), divided by
    pi (1 - pi) as in Lemma 1, and compared with the AUROC lost by transfer.
(3) Sign reversals. For every feature group, the features whose single-feature AUROC lies on opposite
    sides of 0.5 on different backbones (by at least eps), which no per-backbone monotone
    normalization can remove.

  python -m trac.analysis.transfer.diagnostics --tag gsm8k_qwen8b_k8   -> EXP_ROOT/transfer/diagnostics/{tag}.json
  python -m trac.analysis.transfer.diagnostics --aggregate             -> EXP_ROOT/reports/transfer_diagnostics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.heads import stack_fit
from trac.analysis.methods import length_f
from trac.analysis.transfer.common import VIEWS, load_all, rank_norm, sources
from trac.benchmarks import MODEL_KEYS, parse_tag

OUT = C.TRANSFER / "diagnostics"
TOKEN, ANSWERS = "Token probs.", "Intermediate answers"
DIAG_VIEWS = [TOKEN, ANSWERS, "TrAC"]
REF_N = [16, 32, 64, 128]
GROUPS = {  # feature groups of the sign-reversal check
    "token probabilities": lambda B: np.hstack([B["TRACE"], B["ANS"]]),
    "length": length_f,
    "trajectory": lambda B: B["TRAJ_4"],
    "likelihood of a": lambda B: B["TFT"],
    "rival answers": lambda B: B["CON"],
}


def ref_rank(X, R):
    """Empirical CDF of every column of X against a reference set R (NaNs kept)."""
    X, R = np.asarray(X, float), np.asarray(R, float)
    out = np.full_like(X, np.nan)
    for j in range(X.shape[1]):
        r = np.sort(R[np.isfinite(R[:, j]), j])
        ok = np.isfinite(X[:, j])
        if len(r) > 1:
            lo, hi = np.searchsorted(r, X[ok, j], "left"), np.searchsorted(r, X[ok, j], "right")
            out[ok, j] = (lo + hi) / 2 / len(r)
    return out


def run_tag(tag):
    d, m = parse_tag(tag)
    cells = load_all()
    if (d, m) not in cells:
        print(f"[diag] {tag} not eligible")
        return
    src = sources(cells, d, m, "lomdo")
    y = cells[(d, m)]["y"].astype(int)
    same_model = [k for k in cells if k[1] == m]
    out = {"y_mean": float(y.mean()), "n": int(len(y)), "ref": {}}
    rng = np.random.RandomState(0)
    for v in [TOKEN, "TrAC"]:
        f = VIEWS[v]
        pred = stack_fit(
            np.vstack([rank_norm(f(cells[k])) for k in src]), np.concatenate([cells[k]["y"] for k in src]).astype(int)
        )
        X = np.asarray(f(cells[(d, m)]), float)
        X[~np.isfinite(X)] = np.nan
        r = {"all": float(roc_auc_score(y, pred(rank_norm(X))))}
        for n in REF_N:
            if n < len(y):
                r[str(n)] = float(
                    np.mean([roc_auc_score(y, pred(ref_rank(X, X[rng.choice(len(y), n, replace=False)]))) for _ in range(5)])
                )
        R = np.vstack([np.asarray(f(cells[k]), float) for k in same_model])
        R[~np.isfinite(R)] = np.nan
        r["backbone"] = float(roc_auc_score(y, pred(ref_rank(X, R))))
        Ro = np.vstack([np.asarray(f(cells[k]), float) for k in same_model if k[0] != d])
        Ro[~np.isfinite(Ro)] = np.nan
        r["backbone_other"] = float(roc_auc_score(y, pred(ref_rank(X, Ro))))
        pool = {}
        for k in src:
            if k[1] not in pool:
                P = np.vstack([np.asarray(f(cells[q]), float) for q in cells if q[1] == k[1]])
                P[~np.isfinite(P)] = np.nan
                pool[k[1]] = P
        raw = lambda k: np.where(np.isfinite(np.asarray(f(cells[k]), float)), np.asarray(f(cells[k]), float), np.nan)
        pred_p = stack_fit(
            np.vstack([ref_rank(raw(k), pool[k[1]]) for k in src]), np.concatenate([cells[k]["y"] for k in src]).astype(int)
        )
        r["pooled_all"] = float(roc_auc_score(y, pred_p(ref_rank(X, R))))
        out["ref"][v] = r
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{tag}.json", "w"), indent=1)
    print(tag, json.dumps(out["ref"]), flush=True)


def ordinal_shift():
    """Proxy ordinal shift and AUROC lost by transfer, per target and view (no fitting beyond isotonic)."""
    rows = []
    for f in sorted(glob.glob(str(C.TRANSFER / "lomdo_rank" / "*.npz"))):
        tag = os.path.basename(f)[:-4]
        zt, zi = np.load(f), np.load(C.SCORES / f"{tag}.npz")
        y = zt["y"]
        pi = y.mean()
        for norm in ["rank", "none"]:
            zs = np.load(C.TRANSFER / f"lomdo_{norm}" / f"{tag}.npz")
            for v in DIAG_VIEWS:
                sT, sS = zi[v], zs[v]
                g = IsotonicRegression(out_of_bounds="clip").fit(sS, sT).predict(sS)
                shift = float(np.mean(np.abs(sT - g)))
                rows.append(
                    {
                        "tag": tag,
                        "view": v,
                        "norm": norm,
                        "pi": float(pi),
                        "shift": shift,
                        "bound_term": shift / (pi * (1 - pi)),
                        "loss": float(roc_auc_score(y, sT) - roc_auc_score(y, sS)),
                    }
                )
    return rows


def sign_reversals(cells, eps=(0.0, 0.02, 0.05)):
    """Per feature group: features whose backbone-mean single-feature AUROC is above 0.5 + eps on one
    backbone and below 0.5 - eps on another, among the informative features (|AUROC - 0.5| > 0.05 on
    some backbone)."""
    out = {}
    for group, f in GROUPS.items():
        per_model = []
        for m in MODEL_KEYS:
            rows = []
            for (_, mm), B in cells.items():
                if mm != m:
                    continue
                X, y = np.asarray(f(B), float), B["y"]
                a = []
                for j in range(X.shape[1]):
                    x, ok = X[:, j], np.isfinite(X[:, j])
                    informative = ok.sum() > 20 and len(set(y[ok])) == 2 and np.nanstd(x[ok]) > 0
                    a.append(roc_auc_score(y[ok], x[ok]) if informative else np.nan)
                rows.append(a)
            per_model.append(np.nanmean(np.array(rows), 0))
        M = np.array(per_model)
        informative = np.nanmax(np.abs(M - 0.5), 0) > 0.05
        out[group] = {"features": int(M.shape[1]), "informative": int(informative.sum())}
        for e in eps:
            out[group][f"reversed@{e:g}"] = int(((np.nanmax(M, 0) > 0.5 + e) & (np.nanmin(M, 0) < 0.5 - e) & informative).sum())
    return out


def aggregate():
    ref = {}
    for f in sorted(glob.glob(str(OUT / "*.json"))):
        r = json.load(open(f))["ref"]
        for v, d in r.items():
            for k, x in d.items():
                ref.setdefault(v, {}).setdefault(k, []).append(x)
    res = {"ref": {v: {k: (float(np.mean(x)), len(x)) for k, x in d.items()} for v, d in ref.items()}}
    rows = ordinal_shift()
    sh, lo = np.array([r["shift"] for r in rows]), np.array([r["loss"] for r in rows])
    bt = np.array([r["bound_term"] for r in rows])
    res["ordinal"] = {
        "rows": rows,
        "spearman_shift_loss": float(spearmanr(sh, lo).correlation),
        "spearman_bound_loss": float(spearmanr(bt, lo).correlation),
        "bound_holds": float(np.mean(lo <= bt)),
        "by_view": {
            f"{v}/{n}": {
                "shift": float(np.mean([r["shift"] for r in rows if r["view"] == v and r["norm"] == n])),
                "loss": float(np.mean([r["loss"] for r in rows if r["view"] == v and r["norm"] == n])),
            }
            for v in DIAG_VIEWS
            for n in ["rank", "none"]
        },
    }
    vm = res["ordinal"]["by_view"]
    res["ordinal"]["spearman_by_view_means"] = float(
        spearmanr([x["shift"] for x in vm.values()], [x["loss"] for x in vm.values()]).correlation
    )
    res["sign_reversals"] = sign_reversals(load_all())
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(C.REPORTS / "transfer_diagnostics.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("ref", "sign_reversals")}, indent=1))
    print(
        "spearman(shift, loss):",
        round(res["ordinal"]["spearman_shift_loss"], 3),
        "spearman(bound term, loss):",
        round(res["ordinal"]["spearman_bound_loss"], 3),
        "bound holds:",
        res["ordinal"]["bound_holds"],
    )
    for k, x in vm.items():
        print(f"  {k:24s} shift {x['shift']:.4f}  loss {x['loss']:+.3f}")


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
