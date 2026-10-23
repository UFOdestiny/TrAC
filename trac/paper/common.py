"""Shared data access and formatting for the paper's tables and figures."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.cells import load_cell
from trac.analysis.methods import pce_f, raw_baselines, tup_f

TAB = C.PAPER_DIR / "table"
FIG = C.PAPER_DIR / "figures"

MODELS = [("qwen8b", "Qwen3-8B"), ("llama8b", "Llama-3.1-8B"), ("ministral8b", "Ministral-8B")]
DSETS = [("gsm8k", "GSM8K"), ("math500", "MATH500"), ("minerva", "Minerva"), ("olympiad", "Olympiad"),
         ("aime3", "AIME"), ("gpqa_diamond", "GPQA"), ("mmlupro", "MMLU-Pro"), ("triviaqa", "TriviaQA")]
MATH = {"gsm8k", "math500", "minerva", "olympiad", "aime3"}
MIN_ERR = 10          # settings need this many correct and incorrect responses to enter aggregates
SEED = 0


def tag_of(d, m):
    return f"{d}_{m}_k8"


def parts(tag):
    t = tag[:-3]
    for m, _ in MODELS:
        if t.endswith("_" + m):
            return t[: -len(m) - 1], m
    raise ValueError(tag)


def load_cells():
    """cell -> dict(y, s: {method: out-of-fold or raw scores}, B: feature blocks, aux, ok)."""
    cells = {}
    for f in sorted(glob.glob(str(C.SCORES / "*.npz"))):
        tag = os.path.basename(f)[:-4]
        o = dict(np.load(f))
        y = o.pop("y").astype(int)
        B = load_cell(tag)
        sc = dict(o)
        for d in raw_baselines(B).values():
            sc.update({k: v for k, v in d.items() if np.isfinite(v).all()})
        fa = C.AUXILIARY / f"{tag}.npz"
        aux = dict(np.load(fa)) if fa.exists() else {}
        cells[tag] = {"y": y, "s": sc, "B": B, "aux": aux, "ok": min(y.sum(), (1 - y).sum()) >= MIN_ERR}
    return cells


def macro(cells, name, fn, only_ok=True, filt=None):
    v = [fn(c["s"][name], c["y"]) for t, c in cells.items()
         if (c["ok"] or not only_ok) and name in c["s"] and (filt is None or filt(t))]
    return float(np.mean(v)) if v else float("nan")


def auc(s, y):
    return roc_auc_score(y, s)


def boot_ci(cells, name, n=2000):
    """Pair-equal hierarchical bootstrap CI of the macro AUROC."""
    rng = np.random.RandomState(SEED)
    cl = [(c["y"], c["s"][name]) for c in cells.values() if c["ok"] and name in c["s"]]
    vals = []
    for _ in range(n):
        v = []
        for i in rng.choice(len(cl), len(cl), True):
            y, s = cl[i]
            ip, ng = np.where(y == 1)[0], np.where(y == 0)[0]
            ii = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ng, len(ng), True)])
            v.append(roc_auc_score(y[ii], s[ii]))
        vals.append(np.mean(v))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}"


def best_marks(vals, higher=True):
    """bold best / underline second over finite values."""
    fin = [v for v in vals if np.isfinite(v)]
    if not fin:
        return ["--"] * len(vals)
    srt = sorted(set(fin), reverse=higher)
    out = []
    for v in vals:
        s = f3(v)
        if np.isfinite(v) and v == srt[0]:
            s = r"\textbf{" + s + "}"
        elif len(srt) > 1 and np.isfinite(v) and v == srt[1]:
            s = r"\underline{" + s + "}"
        out.append(s)
    return out


def latency_rel():
    """Median relative latency over the audits: SC@k and the endpoint probe from endpoint_*.json,
    the trajectory configurations from trajectory_*.json ('traj_C_S'); 'trac' is the full method."""
    rel = [json.load(open(f))["relative"] for f in glob.glob(str(C.LATENCY / "endpoint_*.json"))]
    out = {k: float(np.median([r[k] for r in rel])) for k in rel[0]}
    tr = [json.load(open(f))["relative"] for f in glob.glob(str(C.LATENCY / "trajectory_*.json"))]
    if tr:
        out.update({k: float(np.median([r[k] for r in tr])) for k in tr[0]})
        out["trac"] = out["full_20_4"]        # every query the supervised features use
        out["trac0"] = out["traj_20_4"]       # the training-free score needs only the checkpoint answers
    return out


def reps(B):
    pce = pce_f(B)                                          # completed-prefix + trajectory
    return {"TUP": tup_f(B), "PCE": pce, "TrAC": np.hstack([pce, tup_f(B)])}


# ----------------------------------------------------------------------------- figures
# One color per method across all figures (Okabe-Ito); TrAC keeps the active-view orange of Figure 1.
COL = {"TrAC": "#D55E00", "TrAC-0": "#E69F00", "CF": "#CC79A7", "TUP": "#0072B2", "SC": "#4D4D4D",
       "HS": "#009E73", "P(True)": "#8C8C8C", "SelfDoubt": "#BDBDBD", "ok": "#009E73", "bad": "#D55E00"}


def _plt():
    """matplotlib configured for the paper (Times via LaTeX, 9 pt, final size)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # rendered at final size, Times via LaTeX, 9 pt text (the caption is 10 pt)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 9,
                         "text.latex.preamble": r"\usepackage{times}", "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.4, "legend.handlelength": 1.8})
    return plt
