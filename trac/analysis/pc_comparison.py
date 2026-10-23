"""Comparison with prefix consistency (Iwase et al., 2026) and within-question discrimination.

Uses EXP_ROOT/prefix_consistency (trac.pipeline.prefix_consistency), trajectory (original
trace), and sample_trajectory (greedy 8-checkpoint trajectories of the K-1 extra samples).

  pooled        macro AUROC over settings on the original response: PC vs TrAC-0 vs TrAC
  within        per-question AUROC averaged over questions whose K = 8 traces are mixed
                (Iwase et al.'s metric), for PC, the greedy 8-checkpoint TrAC-0, and the
                trace's mean token log-probability
  votes         SC@8 vs PC-weighted and trajectory-weighted votes for the original answer
  cost          regenerated tokens relative to the original output tokens

  python -m trac.analysis.pc_comparison   -> EXP_ROOT/prefix_consistency.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import answers as A
from trac import config as C
from trac.analysis.cells import load_cell
from trac.analysis.stats import hboot
from trac.analysis.trajectory_features import greedy_only, thin, trac0_score
from trac.store import read_json_gz


def per_question_auc(groups):
    """Mean AUROC over questions whose traces have both labels (ties count half)."""
    v = [roc_auc_score(y, s) for y, s in groups if 0 < y.sum() < len(y)]
    return float(np.mean(v)) if v else np.nan, len(v)


def main():
    out = {"pooled": {}, "within": {}, "votes": {}, "cost": {}}
    pooled, within, votes = {}, {}, {}
    cost = []
    for f in sorted(glob.glob(str(C.PREFIX_CONSISTENCY / "*.json.gz"))):
        tag = os.path.basename(f)[:-8]
        B = load_cell(tag)
        y = B["y"].astype(int)
        PC = {x["id"]: x["traces"] for x in read_json_gz(f)}
        G = {r["id"]: r for r in read_json_gz(C.GEN / f"{tag}.json.gz")}
        ts_f = C.SAMPLE_TRAJECTORY / f"{tag}.json.gz"
        TS = {x["id"]: {s["k"]: s for s in x["samples"]} for x in read_json_gz(ts_f)} if ts_f.exists() else {}
        R2 = {x["id"]: x for x in read_json_gz(C.TRAJECTORY / f"{tag}.json.gz")}
        ids = [str(i) for i in B["ids"]]
        pc0 = np.array([PC.get(i, {}).get("0", {}).get("pc", np.nan) for i in ids])
        o = dict(np.load(C.SCORES / f"{tag}.npz"))
        if min(y.sum(), (1 - y).sum()) >= 10:
            pooled.setdefault("cells", []).append(tag)
            for k, s in [("Prefix consistency", np.nan_to_num(pc0, nan=0.5)),
                         ("TrAC-0", np.nan_to_num(B["TRAC0"][:, 0])),
                         ("TrAC", o["TrAC"])]:
                pooled.setdefault(k, {})[tag] = (y, s)
        # within-question and votes over all K traces
        gw = {"Prefix consistency": [], "TrAC-0 (greedy, 8 ckpts)": [], "Mean log-prob": []}
        vy, vs = [], {"SC@8": [], "PC-weighted vote": [], "Trajectory-weighted vote": []}
        for i, q in enumerate(ids):
            r, pcq = G.get(q), PC.get(q)
            if r is None or pcq is None or q not in R2:
                continue
            labs = [int(r["correct"])] + [int(s["correct"]) for s in r["samples"]]
            pcs = [pcq.get(str(k), {}).get("pc", np.nan) for k in range(len(labs))]
            u0 = [trac0_score(thin(greedy_only(R2[q]), 8))]
            for k in range(len(r["samples"])):
                x = TS.get(q, {}).get(k)
                u0.append(float(np.mean(x["agree_own"])) if x and x["agree_own"] else np.nan)
            mlp = [float(B["BASE"][i, 0])] + [s.get("meanlp", np.nan) for s in r["samples"]]   # mean token log-prob
            yl = np.array(labs)
            for k, s in [("Prefix consistency", pcs), ("TrAC-0 (greedy, 8 ckpts)", u0), ("Mean log-prob", mlp)]:
                s = np.array(s, float)
                if np.isfinite(s).all():
                    gw[k].append((yl, s))
            for k in range(len(labs)):
                t = pcq.get(str(k))
                if t:
                    cost.append((sum(t["regen_tokens"]), t["prefix_tokens"] / 0.75))
            # votes for the original answer (only where every trace has prefix consistency)
            if not np.isfinite(np.array(pcs, float)).all():
                continue
            same = [1.0] + [float(A.equiv(r, s["answer"], r["answer"])) for s in r["samples"]]
            w_pc = np.array(pcs, float) + 0.01
            w_tr = np.nan_to_num(np.array(u0, float), nan=0.0) + 0.01
            same = np.array(same)
            vy.append(labs[0])
            vs["SC@8"].append(same.mean())
            vs["PC-weighted vote"].append((w_pc * same).sum() / w_pc.sum())
            vs["Trajectory-weighted vote"].append((w_tr * same).sum() / w_tr.sum())
        for k, g in gw.items():
            within.setdefault(k, {})[tag] = g
        vy = np.array(vy)
        if min(vy.sum(), (1 - vy).sum()) >= 10:
            for k, s in vs.items():
                votes.setdefault(k, {})[tag] = (vy, np.array(s))
    cells = pooled.get("cells", [])
    print(f"pooled over {len(cells)} settings (original response):")
    for k in ["Prefix consistency", "TrAC-0", "TrAC"]:
        v = [roc_auc_score(*pooled[k][t]) for t in cells]
        out["pooled"][k] = float(np.mean(v)); print(f"  {k:28s} {np.mean(v):.3f}")
    for a, b in [("TrAC-0", "Prefix consistency"), ("TrAC", "Prefix consistency")]:
        m, lo, hi = hboot([(pooled[a][t][0], pooled[a][t][1], pooled[b][t][1]) for t in cells])
        out["pooled"][f"{a} - PC"] = [m, lo, hi]; print(f"  {a} - PC: {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("within-question AUROC (questions with mixed traces), macro over settings where all three exist:")
    pq = {k: {t: per_question_auc(g) for t, g in d.items()} for k, d in within.items()}
    common = [t for t in pq["Prefix consistency"]
              if all(np.isfinite(pq[k].get(t, (np.nan, 0))[0]) and pq[k][t][1] >= 10 for k in pq)]
    for k in pq:
        vals = [pq[k][t][0] for t in common]
        out["within"][k] = float(np.mean(vals)) if vals else None
        print(f"  {k:28s} {np.mean(vals) if vals else float('nan'):.3f} (settings {len(vals)})")
    out["within"]["settings"] = common
    print("votes for the original answer (macro AUROC, settings with >= 10 errors):")
    for k, d in votes.items():
        v = [roc_auc_score(*d[t]) for t in d]
        out["votes"][k] = float(np.mean(v)); print(f"  {k:28s} {np.mean(v):.3f} (settings {len(v)})")
    c = np.array(cost, float)
    if len(c):
        rel = c[:, 0].sum() / c[:, 1].sum()
        out["cost"]["regen_tokens_over_original"] = float(rel)
        print(f"regenerated tokens / original output tokens: {rel:.3f}")
    json.dump(out, open(C.EXP_ROOT / "prefix_consistency.json", "w"), indent=1, default=float)
    print("saved", C.EXP_ROOT / "prefix_consistency.json")


if __name__ == "__main__":
    main()
