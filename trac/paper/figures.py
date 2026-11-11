"""Figures of the paper, written to PAPER_DIR/figures at their printed size (no graphic scaling).

framework      framework.pdf from the hand-drawn framework.tex (XeLaTeX)
teaser         teaser-{endpoint,trajectory,features}.pdf, the introduction panels
insight        insight-{scatter,cost}.pdf, label-free transfer per setting and AUROC against latency
scale          scale.pdf, raw against rank features under label-free transfer
trajectory     trajectory.pdf, the agreement profile along the trace per backbone
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from trac import backbones as M
from trac import config as C
from trac.analysis.features.trajectory import GRID, agreement_profile
from trac.benchmarks import NEW_BACKBONE
from trac.paper.common import COL, DSETS, FIG, FONT_SIZE, MODELS, _plt, auc, latency_rel, macro, tag_of
from trac.store import read_json_gz

PANEL_WIDTH_CM = 5.15
PANEL_HEIGHT_CM = 6.70
ORANGE = COL["TrAC"]
BLUE = COL["Token probs."]
BLACK = "#000000"


def trim(paths, pad=1.5):
    """Crop vector figures to their visible ink without scaling type or rasterizing. Related panels share
    vertical limits, so their axes and subcaptions align; the crop box is reset first, so rebuilds are
    idempotent."""
    import pymupdf

    docs, boxes = [], []
    for path in paths:
        doc = pymupdf.open(path)
        page = doc[0]
        page.set_cropbox(page.mediabox)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
        yy, xx = np.nonzero(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width) < 250)
        box = pymupdf.Rect(xx.min() / 2 - pad, yy.min() / 2 - pad, (xx.max() + 1) / 2 + pad, (yy.max() + 1) / 2 + pad)
        docs.append(doc)
        boxes.append(box & page.rect)
    top, bottom = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    for path, doc, box in zip(paths, docs, boxes):
        box.y0, box.y1 = top, bottom
        doc[0].set_cropbox(box)
        payload = doc.tobytes(garbage=3, deflate=True)
        doc.close()
        Path(path).write_bytes(payload)


FRAMEWORK_WRAPPER = r"""\documentclass[11pt]{article}
\usepackage[paperwidth=18cm,paperheight=10cm,margin=.5cm]{geometry}
\usepackage{fontspec,amsmath,unicode-math,tikz,xcolor}
\setmainfont{Times New Roman}
\setmathfont{texgyretermes-math.otf}
\usetikzlibrary{calc,arrows.meta}
\pagestyle{empty}
\begin{document}\noindent
\input{FRAMEWORK}
\end{document}
"""


def fig_framework(cells=None):
    """Compile framework.tex with XeLaTeX (real Times New Roman) and crop it to the text width."""
    import pymupdf

    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp)
        wrapper = build / "framework-preview.tex"
        wrapper.write_text(FRAMEWORK_WRAPPER.replace("FRAMEWORK", str(FIG / "framework.tex")), encoding="utf-8")
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build}", str(wrapper)],
            check=True,
            capture_output=True,
        )
        # Crop the oversized TeX page without scaling the embedded fonts or geometry.
        with pymupdf.open(build / "framework-preview.pdf") as doc:
            assert len(doc) == 1
            page = doc[0]
            bounds = pymupdf.Rect()
            for drawing in page.get_drawings():
                bounds |= drawing["rect"]
            for block in page.get_text("blocks"):
                bounds |= pymupdf.Rect(block[:4])
            width = 16 / 2.54 * 72
            assert bounds.width + 1 <= width, bounds
            center = (bounds.x0 + bounds.x1) / 2
            page.set_cropbox(pymupdf.Rect(center - width / 2, bounds.y0 - 0.5, center + width / 2, bounds.y1 + 0.5))
            (FIG / "framework.pdf").write_bytes(doc.tobytes(garbage=3, deflate=True))
    trim([FIG / "framework.pdf"])


def fig_trajectory(cells):
    """Mean agreement profile along the trace for correct and incorrect responses, per backbone."""
    plt = _plt()
    xs = np.array(list(GRID) + [1.0])
    prof = defaultdict(list)
    for tag, c in cells.items():
        if not c["ok"]:
            continue
        R = {x["id"]: x for x in read_json_gz(C.TRAJECTORY / f"{tag}.json.gz")}
        P = np.array([agreement_profile(R[str(q)]) if str(q) in R else [np.nan] * len(xs) for q in c["B"]["ids"]])
        prof[M.from_tag(tag).name].append((np.nanmean(P[c["y"] == 1], 0), np.nanmean(P[c["y"] == 0], 0)))
    order = ["Qwen3-8B", "gpt-oss-20b", "Llama-3.1-8B-Instruct", "Ministral-8B-Instruct-2410"]
    short = {
        "Qwen3-8B": "Qwen3-8B, thinking",
        "gpt-oss-20b": "gpt-oss-20b, thinking",
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
        "Ministral-8B-Instruct-2410": "Ministral-8B-Instruct",
    }
    fig, axs = plt.subplots(1, 4, figsize=(16 / 2.54, 6.0 / 2.54), sharey=True)
    for ax, bb in zip(axs, order):
        v = np.array(prof[bb])
        for j, lab, col, ls in [(0, "correct", COL["ok"], "-"), (1, "incorrect", COL["bad"], "--")]:
            mu, lo, hi = v[:, j].mean(0), np.percentile(v[:, j], 25, 0), np.percentile(v[:, j], 75, 0)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(xs, mu, ls=ls, color=col, label=f"{lab} responses")
        ax.set_title(f"{short[bb]}\n({len(v)} settings)", fontsize=FONT_SIZE, pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(False)
        ax.set_xlabel("Position $t_j/T$")
    axs[0].set_ylabel("Share equal to $a$")
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.1)
    fig.tight_layout(pad=0.4, w_pad=0.6, rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "trajectory.pdf")
    plt.close(fig)
    trim([FIG / "trajectory.pdf"])


def fig_scale(cells=None):
    """Label-free transfer with raw and within-setting rank features (reports/transfer.json) against the
    in-domain classifier of the same view: horizontal dot plot, one row per view."""
    T = json.load(open(C.REPORTS / "transfer.json"))
    views = [("Token probs.", "Token probs."), ("+ early-exit signals", "Token probs. + early-exit signals"), ("TrAC", "TrAC")]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.7 / 2.54, 3.6 / 2.54))
    for i, (_, v) in enumerate(views[::-1]):
        raw, rk, iv = T["lomdo_none"]["macro"][v], T["lomdo_rank"]["macro"][v], macro(cells, v, auc)
        col = COL["TrAC"] if v == "TrAC" else (COL["Token probs."] if "Token" in v else "#555555")
        ax.annotate("", xy=(rk, i), xytext=(raw, i), arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
        ax.plot([raw], [i], "o", mfc="white", mec=col, ms=4)
        ax.plot([rk], [i], "o", color=col, ms=4)
        ax.plot([iv], [i], "|", color="black", ms=8, mew=1.1)
    ax.set_yticks(range(len(views)))
    ax.set_yticklabels([r[0] for r in views[::-1]])
    ax.set_xlabel("Macro AUROC (30 settings)")
    ax.set_xlim(0.69, 0.85)
    ax.plot([], [], "o", mfc="white", mec="black", ms=4, ls="", label="raw")
    ax.plot([], [], "o", color="black", ms=4, ls="", label="rank")
    ax.plot([], [], "|", color="black", ms=8, mew=1.1, ls="", label="in-domain")
    ax.set_ylim(-0.5, len(views) - 0.5)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.97),
        ncol=3,
        handletextpad=0.2,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "scale.pdf")
    plt.close(fig)


def _axis(plt, *, left=0.28, top=0.95):
    fig = plt.figure(figsize=(PANEL_WIDTH_CM / 2.54, PANEL_HEIGHT_CM / 2.54))
    ax = fig.add_axes([left, 0.245, 0.96 - left, top - 0.245])
    ax.tick_params(pad=2, length=2.5, direction="in", colors=BLACK)
    return fig, ax


def _save(plt, fig, name):
    # A common canvas, plot baseline and label baseline align each triptych in the paper.
    fig.set_size_inches(PANEL_WIDTH_CM / 2.54, PANEL_HEIGHT_CM / 2.54)
    ax = fig.axes[0]
    pos = ax.get_position()
    ax.set_position([pos.x0, 0.28, pos.width, 0.50])
    for _ in range(12):
        fig.canvas.draw()
        box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        pos = ax.get_position()
        left = pos.x0 + max(0, 0.025 - box.x0) / fig.get_figwidth()
        right = pos.x1 - max(0, box.x1 - fig.get_figwidth() + 0.025) / fig.get_figwidth()
        if abs(left - pos.x0) + abs(right - pos.x1) < 0.001:
            break
        ax.set_position([left, pos.y0, max(0.12, right - left), pos.height])
    pos = ax.get_position()
    ax.xaxis.set_label_coords((pos.x0 + pos.x1) / 2, 0.28 - 20 / (fig.get_figheight() * 72), transform=fig.transFigure)
    ax.xaxis.label.set_verticalalignment("top")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches=None)
    plt.close(fig)


def fig_teaser(cells):
    """The three introduction panels (teaser-*.pdf): agreement at the end of the trace, agreement along
    the trace, and TrAC without target labels against target labels (reports/few_labels.json)."""
    plt = _plt()
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "axes.edgecolor": BLACK,
        }
    )
    expected = {tag_of(d, m) for d, _ in DSETS for m, _ in MODELS}
    eligible = {t: c for t, c in cells.items() if t in expected and c["ok"]}
    if len(eligible) != 30:
        raise ValueError(f"Expected the paper's 30 eligible settings, got {len(eligible)}")
    for t, c in eligible.items():
        if not np.array_equal(c["y"], c["B"]["y"].astype(int)):
            raise ValueError(f"Feature/score labels differ for {t}")
        if not {"Token probs.", "Intermediate answers", "TrAC", "Supervised prior signals"} <= c["s"].keys():
            raise ValueError(f"Missing main-paper predictions for {t}")
    FIG.mkdir(parents=True, exist_ok=True)

    # Conditional endpoint agreement: equal weight per eligible setting.
    endpoint = []
    for correct in [1, 0]:
        values = []
        for c in eligible.values():
            a = c["B"]["AGREE"][:, 0]
            keep = (c["y"] == correct) & np.isfinite(a)
            if not keep.any():
                raise ValueError("An eligible setting has no finite endpoint agreement in a class")
            values.append(float(a[keep].mean()))
        endpoint.append(float(np.mean(values)))
    fig, ax = _axis(plt, left=0.30, top=0.80)
    for y, value, col in [(1, endpoint[0], COL["ok"]), (0, endpoint[1], COL["bad"])]:
        ax.hlines(y, 0, 100 * value, color=col, lw=1.2)
        ax.plot(100 * value, y, "o", color=col, ms=5)
        ax.annotate(f"{100 * value:.1f}%", (100 * value, y), xytext=(-2, 10), textcoords="offset points", ha="right", color=BLACK)
    ax.set_yticks([1, 0], ["Correct", "Incorrect"])
    ax.set_ylim(-0.55, 1.80)
    ax.text(4, 0.45, "Errors can\nrepeat", va="center", color=BLACK)
    ax.annotate(
        "",
        xy=(100 * endpoint[1] - 2, 0.03),
        xytext=(46, 0.18),
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.7, mutation_scale=5),
    )
    ax.set_xlim(0, 103)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Answer agreement (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(plt, fig, "teaser-endpoint")

    # Class means within a setting, then an equal-weight mean over settings.
    xs = np.array(list(GRID) + [1.0])
    trajectories = []
    for t, c in eligible.items():
        records = {str(r["id"]): r for r in read_json_gz(C.TRAJECTORY / f"{t}.json.gz")}
        ids = [str(q) for q in c["B"]["ids"]]
        if any(q not in records for q in ids):
            raise ValueError(f"Missing trajectory records for {t}")
        profiles = np.asarray([agreement_profile(records[q]) for q in ids], float)
        means = np.asarray([np.nanmean(profiles[c["y"] == v], axis=0) for v in [1, 0]])
        if not np.isfinite(means).all():
            raise ValueError(f"Non-finite mean trajectory for {t}")
        trajectories.append(means)
    trajectory = np.mean(trajectories, axis=0)
    fig, ax = _axis(plt, top=0.80)
    for row, label, col, ls in [(0, "Correct", COL["ok"], "-"), (1, "Incorrect", COL["bad"], "--")]:
        ax.plot(xs, 100 * trajectory[row], color=col, ls=ls, lw=1.5, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Position in reasoning trace")
    ax.set_ylabel("Answer agreement (%)", labelpad=2)
    ax.text(0.57, 89, "Correct", va="center", color=BLACK)
    ax.text(0.43, 12, "Incorrect", va="center", color=BLACK)
    ax.plot(xs[3], 100 * trajectory[0, 3], "o", ms=3.5, color=COL["ok"])
    ax.annotate(
        "Earlier\nagreement",
        xy=(xs[3], 100 * trajectory[0, 3]),
        xytext=(0.04, 66),
        va="center",
        color=BLACK,
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.7, mutation_scale=5),
    )
    _save(plt, fig, "teaser-trajectory")

    # A new model: TrAC trained on n of its labels against TrAC transferred without any (few_labels.json).
    F = json.load(open(C.REPORTS / "few_labels.json"))
    ns = ["50", "100", "250", "all"]
    best = [F[n]["in_lr"] for n in ns]  # the same logistic-regression classifier on both sides
    free = F["50"]["lf_lr"]
    fig, ax = _axis(plt, top=0.80)
    ax.plot(range(4), best, "o-", color="#6f6f6f", ms=4, lw=1.1, label="trained on $n$ target labels")
    ax.axhline(free, color=ORANGE, lw=1.4, ls="--", label="no target label")
    ax.plot([2.1], [free], "*", color=ORANGE, ms=9)
    ax.annotate(f"{free:.3f}"[1:], (0.05, free), xytext=(0, 4), textcoords="offset points", color=ORANGE)
    ax.set_xticks(range(4), ["50", "100", "250", "All"])
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(0.76, 0.87)
    ax.set_yticks([0.78, 0.82, 0.86])
    ax.set_ylabel("Macro AUROC", labelpad=2)
    ax.set_xlabel("Target labels $n$")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.45, 1.0),
        ncol=1,
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.1,
        labelspacing=0.2,
    )
    _save(plt, fig, "teaser-features")
    trim([FIG / f"teaser-{name}.pdf" for name in ("endpoint", "trajectory", "features")])


def fig_insight(cells):
    """Two insight panels for the experiments: (a) per-setting AUROC without target labels, TrAC against the
    strongest transferred baseline, including the held-out Qwen3-32B; (b) AUROC against measured latency."""
    plt = _plt()
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE - 1,
        }
    )
    W, H = 7.6 / 2.54, 5.6 / 2.54
    T = json.load(open(C.REPORTS / "transfer.json"))["lomdo_rank"]["auroc"]
    base, ours = T["Token probs. + early-exit signals"], T["TrAC"]
    NB = json.load(open(C.REPORTS / f"new_backbone_{NEW_BACKBONE}.json"))["per"]
    cols = {"qwen8b": "#24658B", "llama8b": "#238477", "ministral8b": "#805A9B", "gptoss20b": "#AD8426"}
    names = {"qwen8b": "Qwen3-8B", "llama8b": "Llama-3.1-8B", "ministral8b": "Ministral-8B", "gptoss20b": "gpt-oss-20b"}
    fig, ax = plt.subplots(figsize=(W, H))
    for m, col in cols.items():
        ks = [k for k in ours if k.endswith("_" + m)]
        ax.plot([base[k] for k in ks], [ours[k] for k in ks], "o", color=col, ms=3.6, alpha=0.9, label=names[m])
    ax.plot(
        [NB[d]["Token probs. + early-exit signals (LOMDO)"] for d in NB],
        [NB[d]["TrAC (LOMDO)"] for d in NB],
        "*",
        color=ORANGE,
        ms=7,
        label="Qwen3-32B (held out)",
    )
    lo, hi = 0.6, 1.0
    ax.plot([lo, hi], [lo, hi], color=BLACK, lw=0.6, ls=":")
    wins = sum(ours[k] > base[k] for k in ours)
    ax.text(0.62, 0.975, f"TrAC better in {wins} of {len(ours)}\nsettings (circles)", va="top", fontsize=FONT_SIZE - 1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Token probs. + early-exit signals")
    ax.set_ylabel("TrAC without target labels")
    ax.legend(
        loc="lower right",
        frameon=False,
        handletextpad=0.05,
        borderaxespad=0.05,
        labelspacing=0,
        markerscale=0.9,
        fontsize=FONT_SIZE - 2.5,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "insight-scatter.pdf")
    plt.close(fig)

    L = latency_rel()
    ok = {t: c for t, c in cells.items() if c["ok"]}
    sc2 = float(np.mean([auc(c["B"]["SC2"][:, 0], c["y"]) for c in ok.values() if "SC2" in c["B"]]))
    pts = [
        ("Token probs.", 1.0, macro(cells, "Token probs.", auc), BLUE, "s"),
        ("TrAC, minimal", L["trac_min"], macro(cells, "TrAC (minimal)", auc), ORANGE, "o"),
        ("TrAC", L["trac"], macro(cells, "TrAC", auc), ORANGE, "*"),
        ("SC@2", L["sc_2"], sc2, BLACK, "o"),
        ("SC@8", L["sc_8"], macro(cells, "SC@8", auc), BLACK, "o"),
        ("TrAC@8", L["sc_8"] + L["trac"] - 1, macro(cells, "TrAC@8", auc), ORANGE, "*"),
    ]
    fig, ax = plt.subplots(figsize=(W, H))
    off = {
        "Token probs.": (7, 3),
        "TrAC, minimal": (6, -10),
        "TrAC": (5, 4),
        "SC@2": (5, -3),
        "SC@8": (-6, -12),
        "TrAC@8": (-40, 4),
    }
    for lab, x, y, col, mk in pts:
        ax.plot(x, y, mk, color=col, ms=8 if mk == "*" else 4.5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off[lab], color=col, fontsize=FONT_SIZE - 1)
    ax.plot([L["sc_2"], L["sc_8"]], [sc2, macro(cells, "SC@8", auc)], color=BLACK, lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3])
    ax.set_xticklabels(["1", "1.5", "2", "3"])
    ax.minorticks_off()
    ax.set_xlim(0.95, 3.6)
    ax.set_ylim(0.75, 0.90)
    ax.set_xlabel("Latency relative to one trace (log)")
    ax.set_ylabel("Macro AUROC")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "insight-cost.pdf")
    plt.close(fig)
