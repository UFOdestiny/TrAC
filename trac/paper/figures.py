"""Figures of the paper (written to PAPER_DIR/figures)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from trac import backbones as M
from trac import config as C
from trac.analysis.trajectory_features import GRID
from trac.metrics import risk_at_coverage
from trac.paper.common import COL, FIG, _plt, auc, macro
from trac.store import read_json_gz


def figures(cells, eff_rows):
    plt = _plt()
    W, H = 7.7 / 2.54, 5.8 / 2.54
    cov = np.linspace(0.05, 1.0, 40)
    curves = [("\\m (+CF@8)", "TrAC + CF@8", COL["CF"], "-"),
              ("SC@8", "SC@8 (primary)", COL["SC"], "--"),
              ("\\m", "TrAC", COL["TrAC"], "-"),
              ("HS-probe", "HS-probe", COL["HS"], "-."),
              ("\\m-0", "TrAC-0", COL["TrAC-0"], (0, (5, 1.5))),
              ("P(True)", "P(True)", COL["P(True)"], ":"),
              ("SelfDoubt", "SelfDoubt", COL["SelfDoubt"], ":")]
    fig, ax = plt.subplots(figsize=(W, H))
    for lab, n, col, ls in curves:
        r = np.mean([[risk_at_coverage(c["s"][n], c["y"], x) for x in cov] for c in cells.values()
                     if c["ok"] and n in c["s"]], axis=0)
        ax.plot(cov, r, ls=ls, color=col, label=lab.replace("\\m", "TrAC"), lw=1.6 if n in ("TrAC", "TrAC + CF@8") else 1.2,
                zorder=3 if "TrAC" in n else 2)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (macro)")
    ax.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.8, borderaxespad=0.1)
    ax.set_xlim(0, 1.02); ax.set_ylim(0.08, 0.72); ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "risk_coverage.pdf"); plt.close(fig)

    # quality vs latency; arrows add PCE at a matched sampling budget
    fig, ax = plt.subplots(figsize=(W, H))
    pts = {n: (l, a) for n, l, a in eff_rows}
    m = lambda n: macro(cells, n, auc)
    sc = [pts[k] for k in ["SC@2", "SC@4", "SC@8"]]
    ax.plot([p[0] for p in sc], [p[1] for p in sc], "o--", color=COL["SC"], ms=3.5, lw=1.0, zorder=2)
    for k, p in zip(["SC@2", "SC@4", "SC@8"], sc):
        ax.annotate(k, p, textcoords="offset points", xytext=(4, -9), color=COL["SC"], fontsize=8.5)
    pairs = [((1.0, m("TUP")), pts["\\m\\ (20 checkpoints, $S{=}4$)"]),
             ((pts["SC@2"][0], m("TUP + SC@2")), pts["\\m\\ $+$ SC@2 (supervised)"]),
             ((pts["SC@8"][0], m("TUP + CF@8")), pts["\\m\\ (+CF@8)"])]
    for (p0, p1), lab0, lab1 in zip(pairs, ["TUP", "TUP+SC@2", "TUP+SC@8"], ["TrAC", "TrAC+SC@2", "TrAC+SC@8 (CF@8)"]):
        ax.plot(*p0, "s", color=COL["TUP"], ms=4, zorder=3)
        ax.plot(*p1, "*", color=COL["TrAC"], ms=8, zorder=4)
        ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>", color=COL["TrAC"], lw=0.9, shrinkA=3, shrinkB=4,
                                                 mutation_scale=7), zorder=3)
    t0 = pts["\\m-0 (training-free)"]
    ax.plot(*t0, "^", color=COL["TrAC-0"], ms=5, zorder=3)
    lab = [("TUP", pairs[0][0], (4, -3)), ("TrAC", pairs[0][1], (5, -2)), ("TrAC-0", t0, (5, -3)),
           ("TUP+SC@2", pairs[1][0], (5, -7)), ("TrAC+SC@2", pairs[1][1], (-14, 6)),
           ("TUP+SC@8", pairs[2][0], (-50, -7)), ("TrAC+CF@8", pairs[2][1], (-52, 7))]
    for t, p, off in lab:
        col = COL["TUP"] if t.startswith("TUP") else COL["TrAC-0"] if t == "TrAC-0" else COL["TrAC"]
        ax.annotate(t, p, textcoords="offset points", xytext=off, color=col, fontsize=8.5, va="center")
    ax.set_xscale("log"); ax.set_xticks([1, 1.5, 2, 3, 4]); ax.set_xticklabels(["1", "1.5", "2", "3", "4"])
    ax.minorticks_off(); ax.set_xlim(0.95, 4.6); ax.set_ylim(0.70, 0.90)
    ax.set_xlabel("Relative latency (log scale)"); ax.set_ylabel("Macro AUROC"); ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "pareto_quality.pdf"); plt.close(fig)


def _eps_profile(rec):
    """Share of re-elicited answers (greedy and sampled) equal to a at each grid point and the end."""
    T = rec["T"] or 1
    pts = [(c["k"] / T, c) for c in rec["ck"]] + [(1.0, rec["end"] or {})]
    out = []
    for f in list(GRID) + [1.0]:
        c = min(pts, key=lambda p: abs(p[0] - f))[1] or {}
        ag = ([float(c["agree"])] if "agree" in c else []) + [float(x) for x in (c.get("samp") or {}).get("agree", [])]
        out.append(np.mean(ag) if ag else np.nan)
    return out


def fig_trajectory(cells):
    """Mean agreement profile along the trace for correct and incorrect responses, per backbone."""
    plt = _plt()
    xs = np.array(list(GRID) + [1.0])
    prof = defaultdict(list)
    for tag, c in cells.items():
        if not c["ok"]:
            continue
        R = {x["id"]: x for x in read_json_gz(C.TRAJECTORY / f"{tag}.json.gz")}
        P = np.array([_eps_profile(R[str(q)]) if str(q) in R else [np.nan] * len(xs) for q in c["B"]["ids"]])
        prof[M.from_tag(tag).name].append((np.nanmean(P[c["y"] == 1], 0), np.nanmean(P[c["y"] == 0], 0)))
    order = ["Qwen3-8B", "Llama-3.1-8B-Instruct", "Ministral-8B-Instruct-2410"]
    short = {"Qwen3-8B": "Qwen3-8B, thinking", "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
             "Ministral-8B-Instruct-2410": "Ministral-8B-Instruct"}
    fig, axs = plt.subplots(1, 3, figsize=(16 / 2.54, 4.4 / 2.54), sharey=True)
    for ax, bb in zip(axs, order):
        v = np.array(prof[bb])
        for j, lab, col, ls in [(0, "correct", COL["ok"], "-"), (1, "incorrect", COL["bad"], "--")]:
            mu, lo, hi = v[:, j].mean(0), np.percentile(v[:, j], 25, 0), np.percentile(v[:, j], 75, 0)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(xs, mu, ls=ls, color=col, label=f"{lab} responses")
        ax.set_title(f"{short[bb]} ({len(v)} settings)", fontsize=9, pad=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.grid(alpha=0.2, lw=0.5)
        ax.set_xlabel("Checkpoint position $t_j/T$")
    axs[0].set_ylabel("Share equal to $a$")
    axs[2].legend(frameon=False, loc="upper left", borderaxespad=0.1)
    fig.tight_layout(pad=0.4, w_pad=0.6); fig.savefig(FIG / "trajectory.pdf"); plt.close(fig)
    return {bb: (np.array(v)[:, 0].mean(0).round(4).tolist(), np.array(v)[:, 1].mean(0).round(4).tolist())
            for bb, v in prof.items()}
