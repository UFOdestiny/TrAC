"""Control analyses of the stage-3 queue (CPU): decoding matrix, generation seeds, latency.

  decoding   original-trace temperature in {0.7 (main), greedy, 0.2, 1.0} -> greedy
             re-elicitation, and T = 0.7 -> T = 0.7, on MATH500 + Minerva for every backbone
  seeds      main seed + seeds 2, 3 (original trace only) on MATH500 + OlympiadBench
  latency    the relative wall-clock of every audit in EXP_ROOT/latency

Per condition: accuracy, agreement rate, out-of-fold AUROC of a fixed logistic head
(trac.analysis.heads.oof) on the completed-prefix probe ("Endpoint"), on Endpoint + TUP
profile ("Endpoint + TUP"), and with the trajectory added ("PCE", "TrAC"), plus TrAC-0.

  python -m trac.analysis.controls   -> EXP_ROOT/controls.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.cells import load_cell
from trac.analysis.heads import oof

MODELS = ["qwen8b", "llama8b", "ministral8b"]


def cell(tag):
    if not (C.FEATURES / f"{tag}.npz").exists():
        return None
    B = load_cell(tag)
    y = B["y"].astype(int)
    A = np.hstack([B["AGREE"], B["FLL"], B["FCONF"]])
    out = {"n": len(y), "acc": float(y.mean()), "agree": float(np.nanmean(B["AGREE"]))}
    if min(y.sum(), (1 - y).sum()) >= 10:
        reps = [("Endpoint", A), ("Endpoint + TUP", np.hstack([A, B["TRACE"]]))]
        if "TRAJ" in B:
            reps += [("PCE", np.hstack([A, B["TRAJ"]])), ("TrAC", np.hstack([A, B["TRACE"], B["TRAJ"]]))]
            out["TrAC-0"] = float(roc_auc_score(y, np.nan_to_num(B["TRAC0"][:, 0])))
        for name, X in reps:
            s = oof(X, y)
            out[name] = float(roc_auc_score(y, s)) if s is not None else np.nan
    return out


def fmt(r):
    if r is None:
        return "   (missing)"
    return f"acc {r['acc']:.3f}  agree {r['agree']:.3f}  " + "  ".join(
        f"{k} {r.get(k, float('nan')):.3f}" for k in ["Endpoint", "Endpoint + TUP", "PCE", "TrAC", "TrAC-0"])


def main():
    res = {"decoding": {}, "seeds": {}, "latency": {}}
    print("== decoding matrix (original trace -> re-elicitation)")
    for m in MODELS:
        for d in ["math500", "minerva"]:
            rows = {"T0.7 -> greedy (default)": f"{d}_{m}_k8", "greedy -> greedy": f"dm_g_{d}_{m}",
                    "T0.2 -> greedy": f"dm_t02_{d}_{m}", "T1.0 -> greedy": f"dm_t10_{d}_{m}",
                    "T0.7 -> T0.7": f"{d}_{m}_k8_pt0.7"}
            print(f"  {d} / {m}")
            for k, t in rows.items():
                r = cell(t)
                res["decoding"][f"{d}/{m}/{k}"] = r
                print(f"    {k:26s} {fmt(r)}")
    print("== generation seeds")
    for m in MODELS:
        for d in ["math500", "olympiad"]:
            rs = [cell(t) for t in [f"{d}_{m}_k8", f"s2_{d}_{m}", f"s3_{d}_{m}"]]
            rs = [r for r in rs if r is not None]
            agg = {k: (float(np.mean([r[k] for r in rs if k in r])), float(np.std([r[k] for r in rs if k in r])))
                   for k in ["acc", "Endpoint", "Endpoint + TUP", "PCE", "TrAC", "TrAC-0"] if any(k in r for r in rs)}
            res["seeds"][f"{d}/{m}"] = agg
            print(f"  {d:9s} {m:12s} " + "  ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in agg.items()))
    print("== latency (relative to one original trace; summed over 64 queries, batch 16)")
    for f in sorted(glob.glob(str(C.LATENCY / "*.json"))):
        j = json.load(open(f))
        res["latency"][os.path.basename(f)[:-5]] = j["relative"]
        print(f"  {os.path.basename(f)[:-5]:26s} " + "  ".join(f"{k} {v:.3f}x" for k, v in j["relative"].items()))
    json.dump(res, open(C.EXP_ROOT / "controls.json", "w"), indent=1)
    print("saved", C.EXP_ROOT / "controls.json")


if __name__ == "__main__":
    main()
