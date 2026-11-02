"""LaTeX tables of the paper, written to PAPER_DIR/table from the reports and scores under EXP_ROOT."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from trac import config as C
from trac.analysis.heads import fit_fixed
from trac.analysis.methods import METHODS, answer_f, trac_f
from trac.benchmarks import MODEL_KEYS, NEW_BACKBONE
from trac.metrics import aurc, brier, ece, risk_at_coverage
from trac.paper.common import (
    DSETS,
    MATH,
    MODELS,
    TAB,
    auc,
    best_marks,
    f3,
    latency_rel,
    macro,
    parts,
    prefix_consistency_cost,
    tag_of,
)

# ----------------------------------------------------------------------------- main table
MAIN_ROWS = [  # (block, [(method, cost)]); cost None: computed from the latency audits
    (
        "Single trace (grey-box)",
        [
            ("Mean log-prob", "1.00$\\times$"),
            ("Output length", "1.00$\\times$"),
            ("P(True)", "$+1$ fwd"),
            ("SelfDoubt", "$+1$ dec"),
            ("Verbalized top-k", "$+1$ dec"),
            ("Supervised prior signals", "$+1$ f/d"),
            ("Prefix consistency", None),
            ("Token probs.", "1.00$\\times$"),
            ("Token probs. + early-exit signals", "$+105$ dec"),
            ("TrAC (minimal)", None),
            ("TrAC", None),
        ],
    ),
    ("Single trace (white-box)", [("Linear probe", "1.00$\\times$"), ("Wb-S", "1.00$\\times$"), ("TrAC + probes", None)]),
    (
        "Single trace with a reward model (RM) or process reward model (PRM)",
        [("Reward model", "$+1$ RM"), ("TrAC + RM", None), ("Math PRM", "$+1$ PRM"), ("TrAC + PRM", None)],
    ),
    (
        "Eight samples",
        [
            ("SelfCheckGPT@8", None),
            ("KLE@8", None),
            ("Semantic entropy@8", None),
            ("Supervised Consensus@8", None),
            ("CISC@8", None),
            ("SC@8", None),
            ("TrAC@8", None),
        ],
    ),
]
LABELS = {
    "Mean log-prob": "Mean LP",
    "Output length": "Length",
    "Verbalized top-k": "Verb.\\ top-3",
    "Supervised prior signals": r"Sup. prior$^{\dagger}$",
    "Prefix consistency": "Prefix cons.",
    "Token probs. + early-exit signals": r"\quad + early-exit sig.$^{\dagger}$",
    "TrAC (minimal)": r"\textbf{\m}, minimal",
    "TrAC": r"\textbf{\m}",
    "TrAC + probes": r"\textbf{\m\,+probes}",
    "TrAC + RM": r"\textbf{\m\ + RM}",
    "Math PRM": r"Math PRM$^{\S}$",
    "TrAC + PRM": r"\textbf{\m\ + PRM}$^{\S}$",
    "SelfCheckGPT@8": "SelfCheckGPT",
    "KLE@8": "KLE",
    "Semantic entropy@8": "SE",
    "Supervised Consensus@8": r"Sup. cons.$^{\dagger}$",
    "CISC@8": "CISC",
    "SC@8": "SC",
    "TrAC@8": r"\textbf{\m{}@8}",
}
OURS = {"TrAC", "TrAC (minimal)", "TrAC + probes", "TrAC@8", "TrAC + RM", "TrAC + PRM"}
MATH_ONLY = {"Math PRM", "TrAC + PRM"}  # judges of mathematical solutions


def main_cost(name, cost, lat, pc_cost):
    """Cost column of the main table: latency relative to generating the trace, or extra calls."""
    if name == "Prefix consistency":
        return f"$+{100 * pc_cost:.0f}\\%$ tok" if pc_cost else "--"
    if name in ("TrAC", "TrAC + probes"):
        return f"{lat['trac']:.2f}$\\times$"
    if name == "TrAC (minimal)":
        return f"{lat['trac_min']:.2f}$\\times$"
    if name in ("TrAC + RM", "TrAC + PRM"):
        return f"{lat['trac']:.2f}$\\times$ $+1$ {name.split()[-1]}"
    if name == "TrAC@8":
        return f"{lat['sc_8'] + lat['trac'] - 1:.2f}$\\times$"
    return cost or f"{lat['sc_8']:.2f}$\\times$"


def table_main(cells):
    """Task-grouped metrics at the document's small font size, without scaling."""
    lat, pc_cost = latency_rel(), prefix_consistency_cost(cells)
    groups = [("Math", lambda t: parts(t)[0] in MATH), ("QA", lambda t: parts(t)[0] not in MATH), ("Overall", None)]
    metrics = [
        (auc, True, False),
        (aurc, False, False),
        (brier, False, True),
        (lambda s, y: risk_at_coverage(s, y, 0.1), False, False),
    ]
    lines = []
    for block, rows in MAIN_ROWS:
        names = [name for name, _ in rows]
        columns = []
        for gname, filt in groups:
            for fn, higher, probabilistic in metrics:
                vals = [
                    macro(cells, n, fn, filt=filt)
                    if (not probabilistic or n in METHODS) and (gname == "Math" or n not in MATH_ONLY)
                    else np.nan
                    for n in names
                ]
                columns.append(best_marks(vals, higher=higher))
        lines.append(r"\multicolumn{14}{@{}l}{\textit{" + block + r"}} \\")
        for i, (name, cost) in enumerate(rows):
            pre = r"\rowcolor{gray!15}" if name in OURS else ""
            values = " & ".join(col[i] for col in columns)
            lines.append(f"{pre}{LABELS.get(name, name)} & {main_cost(name, cost, lat, pc_cost)} & {values} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    head = " & ".join([r"AUROC$\uparrow$ & AURC$\downarrow$ & Brier$\downarrow$ & Risk$\downarrow$"] * 3)
    tex = (
        r"""\begin{table*}[t]
\centering
\small
\caption{\m\ is the strongest single-trace grey-box estimator and improves every signal it is combined with. Macro averages over the five mathematical (Math) and three question-answering (QA) benchmarks; best per block in \textbf{bold}, second \underline{underlined}. Cost: latency relative to generating the trace (Appendix Table~\ref{tab:cost}) or extra calls (fwd: forward pass, dec: short decode, tok: tokens). Mean LP: mean token log-probability; Risk: error rate at 10\% coverage; Sup.: supervised; $\dagger$: our controls with the same classifier; $\S$: mathematical benchmarks only.}
\label{tab:main}
\setlength{\tabcolsep}{1.1pt}
\begin{tabular}{@{}lc*{12}{c}@{}}
\toprule
\multirow{2}{*}{Method} & \multirow{2}{*}{Cost}
& \multicolumn{4}{c}{\textbf{Math}} & \multicolumn{4}{c}{\textbf{QA}} & \multicolumn{4}{c}{\textbf{Overall}} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(l){11-14}
& & """
        + head
        + r""" \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\end{tabular}
\end{table*}
"""
    )
    (TAB / "result.tex").write_text(tex)


# ----------------------------------------------------------------------------- ablation
ABL = [
    ("(a) Components", [("Token probs.", "Token probs."), ("\\m\\ (logistic)", "Abl: logistic head"), ("\\m", "TrAC")]),
    (
        "(b) Queries of \\m",
        [
            ("without rival answers", "Abl: without rival answers"),
            ("rival at the end of the trace only", "Abl: rival at the end only"),
            ("without likelihood of $a$", "Abl: without likelihood of a"),
            ("end of trace only", "Abl: end of trace only"),
            ("greedy answers only", "Abl: greedy answers only"),
            ("20 checkpoints", "Abl: 20 checkpoints"),
            ("\\m", "TrAC"),
        ],
    ),
    (
        "(c) Token probs.\\ plus",
        [
            ("early-exit signals", "Token probs. + early-exit signals"),
            ("likelihood of $a$", "Token probs. + likelihood of a"),
            ("intermediate answers", "Token probs. + intermediate answers"),
            ("verbalized top-3 answers", "Token probs. + verbalized top-k"),
            ("rival answers", "Token probs. + rival answers"),
            ("all (\\m)", "TrAC"),
        ],
    ),
]


def _abl_risk_lines(cells, groups):
    """AUROC, AURC, and risk at 10% coverage; the best value of each metric per group in bold."""
    metrics = [(auc, True), (aurc, False), (lambda s, y: risk_at_coverage(s, y, 0.1), False)]
    lines = []
    for title, rows in groups:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        cols = []
        for fn, higher in metrics:
            vals = [macro(cells, n, fn) for _, n in rows]
            best = max(vals) if higher else min(vals)
            cols.append([r"\textbf{" + f3(v) + "}" if v == best else f3(v) for v in vals])
        for i, (label, _) in enumerate(rows):
            lines.append(f"{label} & " + " & ".join(c[i] for c in cols) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    return lines


def table_ablation(cells):
    lines = _abl_risk_lines(cells, ABL)
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{Every query adds to token-probability features, and the rival query adds the most. Macro averages over 30 settings with the same folds. (a) Classifier; (b) \m\ with one query family removed or changed; (c) token-probability features with one feature group added (early-exit signals: answer convergence point, Certaindex-style late agreement, and DEER end confidence from the same intermediate answers). Risk: error rate at 10\% coverage.}
\label{tab:ablation}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
\textbf{Representation} & \textbf{AUROC$\uparrow$} & \textbf{AURC$\downarrow$} & \textbf{Risk$\downarrow$} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\end{tabular}
\end{table}
"""
    )
    (TAB / "ablation.tex").write_text(tex)


def report(name):
    return json.load(open(C.REPORTS / f"{name}.json"))


def table_transfer(cells):
    """Transfer protocols (trac.analysis.transfer.evaluate): macro AUROC of the views under
    leave-one-model-and-dataset-out with raw and rank features, leave-one-model-and-family-out,
    leave-one-model-out and leave-one-dataset-out with rank features, and in-domain fitting."""
    T = report("transfer")
    cols = [
        ("lomdo_none", "LOMDO raw"),
        ("lomdo_rank", "LOMDO"),
        ("lomfo_rank", "LOMFO"),
        ("lomo_rank", "LOMO"),
        ("lodo_rank", "LODO"),
    ]
    lines = []
    for v, lab in [
        ("Token probs.", "Token probs."),
        ("Token probs. + early-exit signals", r"\quad $+$ early-exit signals"),
        ("TrAC", r"\m"),
        ("TrAC (minimal)", r"\m, minimal"),
    ]:
        vals = [T[k]["macro"].get(v, np.nan) for k, _ in cols] + [macro(cells, v, auc)]
        lines.append(lab + " & " + " & ".join("--" if not np.isfinite(x) else f3(x) for x in vals) + r" \\")
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{\m\ leads under every transfer protocol. Macro AUROC over 30 settings when the target's model and dataset (LOMDO), model and task family (LOMFO), model (LOMO), or dataset (LODO) are held out; rank features unless marked raw}
\label{tab:transfer}
\setlength{\tabcolsep}{2.4pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \textbf{LOMDO raw} & \textbf{LOMDO} & \textbf{LOMFO} & \textbf{LOMO} & \textbf{LODO} & \textbf{In-domain} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    )
    (TAB / "transfer.tex").write_text(tex)


def table_label_eff(cells):
    """Fixed 40% test split; label fractions of the remaining 60%. All fractions use the same
    cells (those whose smallest labeled subset still holds 3 examples per class in every repeat)
    and a logistic regression, so rows are directly comparable."""
    from sklearn.model_selection import StratifiedShuffleSplit

    fr = [0.05, 0.10, 0.25, 0.50, 1.0]
    plan = {}
    for t, c in cells.items():
        if not c["ok"]:
            continue
        y = c["y"]
        ok = True
        sp = []
        for rep in range(3):
            tr0, te = next(StratifiedShuffleSplit(1, test_size=0.4, random_state=rep).split(y.reshape(-1, 1), y))
            perm = np.random.RandomState(rep).permutation(tr0)
            idx = {f: (perm[: int(round(f * len(tr0)))] if f < 1 else tr0) for f in fr}
            if min(y[idx[fr[0]]].sum(), (1 - y[idx[fr[0]]]).sum()) < 3:
                ok = False
                break
            sp.append((idx, te))
        if ok:
            plan[t] = sp
    res = defaultdict(lambda: defaultdict(list))
    views = {"Intermediate answers": answer_f, "TrAC": trac_f}
    for t, sp in plan.items():
        B, y = cells[t]["B"], cells[t]["y"]
        for idx, te in sp:
            for f in fr:
                for r, view in views.items():
                    X = np.asarray(view(B), float)
                    m = fit_fixed("lr", X[idx[f]], y[idx[f]])
                    res[r][f].append(auc(m.predict_proba(X[te])[:, 1], y[te]))
    full = np.mean(res["TrAC"][1.0])
    lines = [
        f"{f * 100:g}\\% & {f3(np.mean(res['Intermediate answers'][f]))} & {f3(np.mean(res['TrAC'][f]))} & "
        f"{100 * np.mean(res['TrAC'][f]) / full:.1f}\\% \\\\"
        for f in fr
    ]
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{Label efficiency on a fixed 40\% test split (three repeats). Only the labeled fraction of the
remaining 60\% changes. All rows use the same """
        + str(len(plan))
        + r""" settings and logistic regression;
retention is relative to \m\ with all labels.}
\label{tab:label-efficiency}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
\textbf{Labels} & \shortstack{Intermediate\\answers} & \textbf{\m} & \textbf{Retention} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    (TAB / "label_efficiency.tex").write_text(tex)


# ----------------------------------------------------------------------------- appendix tables
def app_matrix(cells):
    single = ["SelfDoubt", "Token probs.", "TrAC"]
    multi = ["SC@8", "TrAC@8"]
    lines, per_model = [], defaultdict(list)
    for d, dn in DSETS:
        rows = [(m, mn, tag_of(d, m)) for m, mn in MODELS if tag_of(d, m) in cells]
        for i, (m, mn, t) in enumerate(rows):
            c = cells[t]
            y = c["y"]
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{dn}}}" if i == 0 else ""
            if c["ok"]:
                v1 = [auc(c["s"][n], y) for n in single]
                v2 = [auc(c["s"][n], y) for n in multi]
                per_model[mn].append(v1 + v2)
                vs = best_marks(v1) + best_marks(v2)
            else:
                vs = ["--"] * (len(single) + len(multi))
            lines.append(f"{first} & {mn} & {len(y)} & {f3(y.mean())} & " + " & ".join(vs) + " \\\\")
        lines.append("\\midrule")
    for k, (m, mn) in enumerate(MODELS):
        v = np.mean(per_model[mn], 0)
        first = f"\\multirow{{{len(MODELS)}}}{{*}}{{Macro}}" if k == 0 else ""
        lines.append(
            f"{first} & {mn} & & & "
            + " & ".join(best_marks(list(v[: len(single)])) + best_marks(list(v[len(single) :])))
            + " \\\\"
        )
    lines.append("\\bottomrule")
    tex = (
        r"""\begin{table*}[t]
\centering
\small
\caption{AUROC for every setting, with macro averages per backbone; AIME for Llama-3.1-8B and Ministral-8B has fewer than ten correct responses and is excluded from aggregates}
\label{tab:app-matrix}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccccc}
\toprule
\textbf{Dataset} & \textbf{Model} & $n$ & \textbf{Acc.} & \textbf{SelfDoubt} & \shortstack{Token\\probs.} & \textbf{\m} & \textbf{SC@8} & \textbf{\m{}@8} \\\\
\midrule
"""
        + "\n".join(lines)
        + r"""
\end{tabular}
\end{table*}
"""
    )
    (TAB / "app_matrix.tex").write_text(tex)


BLINDSPOT = ["SC@8", "Token probs.", "Intermediate answers", "TrAC", "Linear probe"]


def app_blindspot(cells):
    lines = []
    for th in [0.625, 0.75, 0.875, 1.0]:
        n = w = 0
        v = defaultdict(list)
        for c in cells.values():
            if not c["ok"]:
                continue
            m = c["s"]["SC@8"] >= th - 1e-9
            y = c["y"][m]
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            n += m.sum()
            w += (1 - y).sum()
            for k in BLINDSPOT:
                s = c["s"][k][m]
                v[k].append(auc(s, y) if np.ptp(s) > 0 else 0.5)
        lines.append(
            f"{th:.3f} & {len(v['SC@8'])} & {n:,} & {w:,} & " + " & ".join(f3(np.mean(v[k])) for k in BLINDSPOT) + " \\\\"
        )
    tex = (
        r"""\begin{table*}[t]
\centering
\small
\caption{\m\ still separates errors when all samples agree. AUROC on subsets where the final answer's vote share among eight samples reaches the threshold; Wrong: incorrect responses}
\label{tab:app-blindspot}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{ccccccccc}
\toprule
\textbf{Vote $\geq$} & \textbf{Settings} & $n$ & \textbf{Wrong} & \textbf{SC} & \shortstack{Token\\probs.} & \shortstack{Intermediate\\answers} & \textbf{\m} & \textbf{Probe} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    (TAB / "app_blindspot.tex").write_text(tex)


def app_fusion(cells):
    lines = []
    for d, dn in DSETS:
        for m, mn in MODELS:
            t = tag_of(d, m)
            if t not in cells or not cells[t]["ok"]:
                continue
            c = cells[t]
            v = [auc(c["s"][n], c["y"]) for n in ["SC@8", "Semantic entropy@8", "CISC@8", "TrAC@8"]]
            lines.append(f"{dn} & {mn} & " + " & ".join(f3(x) for x in v) + f" & ${v[3] - v[0]:+.3f}$ \\\\")
    tex = (
        r"""\begin{table*}[t]
\centering
\small
\caption{Combining \m with eight-sample agreement against eight-sample baselines, per setting.}
\label{tab:app-fusion}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llccccc}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{SC@8} & \textbf{SE@8} & \textbf{CISC@8} & \textbf{\m{}@8} & $\boldsymbol{\Delta}$ \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    (TAB / "app_fusion.tex").write_text(tex)


def app_leakage_length(cells):
    sub = defaultdict(lambda: defaultdict(list))
    for t, c in cells.items():
        if not c["ok"] or parts(t)[0] not in MATH or "in_trace" not in c["aux"]:
            continue
        f = c["aux"]["in_trace"].astype(bool)
        for lab, msk in [("stated", f), ("not stated", ~f)]:
            y = c["y"][msk]
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            for k in ["Intermediate answers", "TrAC"]:
                sub[lab][k].append(auc(c["s"][k][msk], y))
            sub[lab]["frac"].append(msk.mean())
    allf = np.concatenate(
        [c["aux"]["in_trace"] for t, c in cells.items() if c["ok"] and parts(t)[0] in MATH and "in_trace" in c["aux"]]
    )
    share = {"stated": allf.mean(), "not stated": 1 - allf.mean()}
    lines = [
        f"Answer {lab} in trace & {len(sub[lab]['TrAC'])} & {100 * share[lab]:.0f}\\% & "
        f"{f3(np.mean(sub[lab]['Intermediate answers']))} & \\textbf{{{f3(np.mean(sub[lab]['TrAC']))}}} \\\\"
        for lab in ["stated", "not stated"]
    ]
    (TAB / "app_leakage.tex").write_text(
        r"""\begin{table}[t]
\centering
\small
\caption{\m\ stays accurate when the trace never writes the final answer. Mathematical settings split by whether the reasoning contains the answer string; Share: fraction of responses}
\label{tab:app-leakage}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.37\columnwidth}cccc@{}}
\toprule
\textbf{Subset} & \textbf{Settings} & \textbf{Share} & \shortstack{Intermediate\\answers} & \textbf{\m} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    ter = defaultdict(lambda: defaultdict(list))
    for t, c in cells.items():
        if not c["ok"] or "ntok" not in c["aux"]:
            continue
        nt = c["aux"]["ntok"]
        q = np.quantile(nt, [1 / 3, 2 / 3])
        for i, msk in enumerate([nt <= q[0], (nt > q[0]) & (nt <= q[1]), nt > q[1]]):
            y = c["y"][msk]
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            for k in ["Token probs.", "Intermediate answers", "TrAC"]:
                ter[i][k].append(auc(c["s"][k][msk], y))
    lines = [
        f"{lab} & {len(ter[i]['TrAC'])} & {f3(np.mean(ter[i]['Token probs.']))} & {f3(np.mean(ter[i]['Intermediate answers']))} & "
        f"\\textbf{{{f3(np.mean(ter[i]['TrAC']))}}} \\\\"
        for i, lab in enumerate(["Short", "Medium", "Long"])
    ]
    (TAB / "app_length.tex").write_text(
        r"""\begin{table}[t]
\centering
\small
\caption{Length control. AUROC within per-setting output-length tertiles (macro over settings with at least five
correct and five incorrect responses in the tertile).}
\label{tab:app-length}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
\textbf{Length tertile} & \textbf{Settings} & \shortstack{Token\\probs.} & \shortstack{Intermediate\\answers} & \textbf{\m} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )


def app_taxonomy(cells):
    cnt = defaultdict(lambda: np.zeros(4))
    for t, c in cells.items():
        if "tax" not in c["aux"]:
            continue
        m = parts(t)[1]
        tax = c["aux"]["tax"]
        for k in range(4):
            cnt[m][k] += (tax == k).sum()
            cnt["all"][k] += (tax == k).sum()
    lines = []
    for m, mn in MODELS + [("all", "All")]:
        v = cnt[m]
        s = v.sum()
        lines.append(f"{mn} & {int(s):,} & " + " & ".join(f"{100 * x / s:.1f}" for x in v) + " \\\\")
    (TAB / "app_taxonomy.tex").write_text(
        r"""\begin{table*}[t]
\centering
\small
\caption{Models reproduce most of their errors at the end of the trace. What the intermediate answer at the end returns when the original answer is wrong (\% of incorrect responses pooled over all 32
settings): the correct answer, the same wrong answer, a different wrong answer, or no parsable answer.}
\label{tab:app-taxonomy}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Wrong} & \textbf{Correct} & \textbf{Same wrong} & \textbf{Diff.\ wrong} & \textbf{Unparsable} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )


def app_dataset(cells):
    lines = []
    reg = {
        "gsm8k": "Grade-school word problems",
        "math500": "Competition mathematics",
        "minerva": "Technical quantitative",
        "olympiad": "Olympiad-level problems",
        "aime3": "AIME 2024--2026",
        "gpqa_diamond": "Graduate-level science (multiple choice)",
        "mmlupro": "Multitask knowledge (multiple choice, 10 options)",
        "triviaqa": "Open-domain factual QA",
    }
    for d, dn in DSETS:
        ts = [t for t in cells if parts(t)[0] == d]
        n = len(cells[ts[0]]["y"])
        acc = ", ".join(f"{cells[tag_of(d, m)]['y'].mean():.2f}" for m, _ in MODELS if tag_of(d, m) in cells)
        lines.append(f"{dn} & {n} & {acc} & {reg[d]} \\\\")
    (TAB / "app_dataset.tex").write_text(
        r"""\begin{table*}[t]
\centering
\small
\caption{Benchmarks. Acc.\ lists final-answer accuracy for Qwen3-8B, Llama-3.1-8B, Ministral-8B, and gpt-oss-20b.}
\label{tab:app-dataset}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccl}
\toprule
\textbf{Benchmark} & \textbf{Problems} & \textbf{Acc.} & \textbf{Task} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )


def app_calibration(cells):
    """In-domain ECE, Brier score, and NLL of the learned methods, and abstention at a fixed threshold."""
    ok = {t: c for t, c in cells.items() if c["ok"]}
    nll = lambda s, y: float(-np.mean(y * np.log(np.clip(s, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - s, 1e-6, 1))))
    learned = [
        ("Token probs.", "Token probs."),
        ("Prior signals (ours)", "Supervised prior signals"),
        ("\\m", "TrAC"),
        ("Linear probe", "Linear probe"),
        ("Wb-S", "Wb-S"),
        ("\\m\\ + probes", "TrAC + probes"),
        ("Consensus (ours)", "Supervised Consensus@8"),
        ("\\m{}@8", "TrAC@8"),
    ]
    rows = []
    for lab, n in learned:
        v = [(ece(c["s"][n], c["y"]), brier(c["s"][n], c["y"]), nll(c["s"][n], c["y"])) for c in ok.values() if n in c["s"]]
        rows.append(f"{lab} & In-domain & " + " & ".join(f3(x) for x in np.mean(v, 0)) + " & -- \\\\")
    ab = []  # accept responses whose TrAC score reaches tau
    for tau in [0.5, 0.7, 0.9]:
        cov, risk, gap = [], [], []
        for c in ok.values():
            s, y = c["s"]["TrAC"], c["y"]
            m = s >= tau
            cov.append(m.mean())
            if m.sum():
                risk.append(1 - y[m].mean())
                gap.append(1 - s[m].mean())
        ab.append(f"{tau:.1f} & In-domain & {100 * np.mean(cov):.1f}\\% & {f3(np.mean(risk))} & {f3(np.mean(gap))} \\\\")
    tex = (
        r"""\begin{table*}[t]
\centering
\small
\caption{Trained in-domain, \m\ lowers Brier score and NLL relative to token-probability features, and \m{}@8 gives the lowest overall. Top: ECE, Brier score, and negative log-likelihood (NLL); bottom: abstention at a fixed threshold $\tau$, with coverage, error rate, and predicted error rate}
\label{tab:app-calibration}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llcccc}
\toprule
\textbf{Method} & \textbf{Fit} & \textbf{ECE} & \textbf{Brier} & \textbf{NLL} & \\\\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{llccc}
\toprule
$\tau$ & \textbf{Fit} & \textbf{Coverage} & \textbf{Risk} & \textbf{Predicted} \\\\
\midrule
"""
        + "\n".join(ab)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    (TAB / "app_calibration.tex").write_text(tex)


# ----------------------------------------------------------------------------- paired cost audit
def table_cost(cells=None):
    """Paired cost audit (trac.pipeline.throughput): per backbone and concurrency, the median
    latency of every TrAC query relative to generating the original traces (warm prefix cache and
    after a cache reset), SC@8, and the query and token counts per response."""
    rows = []
    for m, mn in MODELS:
        f = C.LATENCY / f"throughput_math500_{m}.json"
        if not f.exists():
            continue
        runs = json.load(open(f))["runs"]
        for B in sorted({r["concurrency"] for r in runs}):
            rs = [r for r in runs if r["concurrency"] == B]
            med = lambda k: float(np.median([(r["primary"] + r[k]) / r["primary"] for r in rs]))
            per = lambda k: float(np.sum([r[k] for r in rs]) / np.sum([r["size"] for r in rs]))
            v = {
                "trac": med("trac"),
                "cold": med("trac_cold"),
                "sc8": med("sc_extra_7"),
                "gen_tok": per("gen_tokens"),
                "queries": per("n_queries"),
            }
            rows.append(
                f"{mn if B == min(r['concurrency'] for r in runs) else ''} & {B} & {v['gen_tok']:.0f} & "
                f"{v['queries']:.0f} & {v['trac']:.3f}$\\times$ & {v['cold']:.3f}$\\times$ & "
                f"{v['sc8']:.2f}$\\times$ \\\\"
            )
    if not rows:
        return
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{With the trace cached, all \m\ queries cost a small fraction of generating the trace. Latency relative to generation on the same fresh MATH500 traces (one H200, vLLM); Conc.: concurrent prompts; Tok.: tokens per trace; Q: queries per response}
\label{tab:cost}
\setlength{\tabcolsep}{2.2pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrccc}
\toprule
Backbone & Conc. & Tok. & Q & \m & \m, cold & SC@8 \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    )
    (TAB / "cost.tex").write_text(tex)


# ----------------------------------------------------------------------------- label-free transfer
def table_transfer_free(cells):
    """Label-free transfer (trac.analysis.transfer): macro AUROC per backbone of training-free scores, of
    classifiers fitted on other backbones' other datasets with rank (and raw) features, of the transferred
    hidden-state probe and text judges, and of the in-domain classifiers for reference."""
    T = report("transfer")
    R, N = T["lomdo_rank"]["auroc"], T["lomdo_none"]["auroc"]
    J, P, V = report("judge_transfer")["per"], report("probe_transfer")["per"], report("verifier_transfer")["per"]

    def row(label, per, key=None, bold=False, shade=False):
        """per: {setting: AUROC, or {key: AUROC}}; mean per backbone and overall."""
        a = {t: (x[key] if key else x) for t, x in per.items()}
        v = [np.mean([x for t, x in a.items() if m in t]) for m in MODEL_KEYS] + [np.mean(list(a.values()))]
        f = (lambda x: r"\textbf{" + f3(x) + "}") if bold else f3
        return (r"\rowcolor{gray!15}" if shade else "") + label + " & " + " & ".join(f(x) for x in v) + r" \\"

    ind = {n: {t: auc(c["s"][n], c["y"]) for t, c in cells.items() if c["ok"] and n in c["s"]} for n in ["Token probs.", "TrAC"]}
    lines = [
        r"\multicolumn{6}{l}{\textit{Training-free}} \\",
        row("Mean LP", R["Mean log-prob"]),
        row("P(True)", R["P(True)"]),
        row("SelfDoubt", R["SelfDoubt"]),
        row("Reward model", R["Reward model"]),
        row("Output length", R["Output length"]),
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Transferred, raw features}} \\",
        row("Token probs.", N["Token probs."]),
        row(r"\m", N["TrAC"]),
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Transferred, rank features}} \\",
        row("Token probs.", R["Token probs."]),
        row(r"\quad $+$ early-exit signals", R["Token probs. + early-exit signals"]),
        row(r"\quad $+$ likelihood of $a$", R["Token probs. + likelihood of a"]),
        row(r"\textbf{\m}", R["TrAC"], bold=True, shade=True),
        row(r"\m, minimal", R["TrAC (minimal)"]),
        row(r"\m\ $+$ RM", R["TrAC + RM"]),
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Transferred hidden-state probe and text-only judges}} \\",
        row("Hidden-state probe", P),
        row("Judge classifier", J, "judge"),
        row(r"\m\ $+$ judge", J, "trac_judge"),
        row("Verifier, answer", V, "Verifier, answer"),
        row("Verifier, full trace", V, "Verifier, full trace"),
        row(r"\m\ $+$ verifier", V, "TrAC + verifier"),
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{In-domain (reference)}} \\",
        row("Token probs.", ind["Token probs."]),
        row(r"\m", ind["TrAC"]),
    ]
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{Without labels of the new model, \m\ transfers better than token-probability features, hidden-state probes, and text judges. Macro AUROC per backbone; classifiers are fitted on the other backbones for the other datasets and use rank features (raw: unranked); each $+$ row adds one feature group to token-probability features; the probe is aligned across backbones by canonical correlation, and $+$ verifier averages the ranks of \m\ and the full-trace verifier.}
\label{tab:transfer-free}
\setlength{\tabcolsep}{2.6pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Qwen3} & \textbf{Llama} & \textbf{Ministral} & \textbf{gpt-oss} & \textbf{All} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    )
    (TAB / "transfer_free.tex").write_text(tex)


# ----------------------------------------------------------------------------- paired comparisons
COMPARISONS = [("in_domain", "In-domain"), ("label_free", "Label-free (LOMDO, rank)")]
PAIR_LABELS = {  # pairs of trac.analysis.evaluation.pairs, in its order
    "in_domain": {
        "TrAC - Token probs.": r"\m\ $-$ token probs.",
        "TrAC - Token probs. + early-exit signals": r"\m\ $-$ token probs.\ + early-exit signals",
        "TrAC - Token probs. + intermediate answers": r"\m\ $-$ token probs.\ + intermediate answers",
        "TrAC - Supervised prior signals": r"\m\ $-$ supervised prior signals",
        "TrAC - Output length": r"\m\ $-$ output length",
        "TrAC - Reward model": r"\m\ $-$ reward model",
        "TrAC - Abl: without rival answers": r"\m\ $-$ \m\ without rival answers",
        "Token probs. + rival answers - Token probs.": r"token probs.\ + rival answers $-$ token probs.",
        "Token probs. + rival answers - Token probs. + verbalized top-k": r"token probs.\ + rival $-$ token probs.\ + verbalized top-3",
        "TrAC - Abl: rival at the end only": r"\m\ $-$ \m\ with the rival at the end only",
        "TrAC - Abl: end of trace only": r"\m\ $-$ \m\ at the end of the trace only",
        "TrAC - Abl: logistic head": r"\m\ $-$ \m\ with a logistic head",
        "TrAC + probes - Wb-S": r"\m\ + probes $-$ Wb-S",
        "TrAC + probes - Token probs. + probes": r"\m\ + probes $-$ token probs.\ + probes",
        "TrAC + RM - Token probs. + RM": r"\m\ + RM $-$ token probs.\ + RM",
        "TrAC - Math PRM": r"\m\ $-$ math PRM (mathematics)",
        "TrAC + PRM - Token probs. + PRM": r"\m\ + PRM $-$ token probs.\ + PRM (mathematics)",
        "TrAC - TrAC (minimal)": r"\m\ $-$ minimal \m",
        "TrAC@8 - SC@8": r"\m{}@8 $-$ SC@8",
        "TrAC@8 - Token probs. + consensus@8": r"\m{}@8 $-$ token probs.\ + consensus",
    },
    "label_free": {
        "TrAC - Token probs.": r"\m\ $-$ token probs.",
        "TrAC - Token probs. + early-exit signals": r"\m\ $-$ token probs.\ + early-exit signals",
        "TrAC - Output length": r"\m\ $-$ output length",
        "TrAC - Reward model": r"\m\ $-$ reward model",
        "TrAC - TrAC (raw)": r"\m\ $-$ \m\ with raw features",
        "TrAC - Judge classifier": r"\m\ $-$ transferred judge classifier",
        "TrAC + RM - Token probs. + RM": r"\m\ + RM $-$ token probs.\ + RM",
    },
}


def app_comparisons(cells=None):
    P = report("pairs")
    lines = []
    for key, title in COMPARISONS:
        lines.append(r"\multicolumn{3}{l}{\textit{" + title + r"}} \\")
        for pair, v in P[key].items():
            lines.append(
                f"{PAIR_LABELS[key][pair]} & ${v['diff']:+.3f}$ $[{v['lo']:+.3f}, {v['hi']:+.3f}]$ & {v['wins']}/{v['n']} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{Paired macro-AUROC differences with 95\% hierarchical-bootstrap intervals, and the number of settings where the first method is better.}
\label{tab:app-comparisons}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcc}
\toprule
\textbf{Comparison} & \textbf{$\Delta$ AUROC [95\% CI]} & \textbf{Wins} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\end{tabular}}
\end{table}
"""
    )
    (TAB / "app_comparisons.tex").write_text(tex)


# ----------------------------------------------------------------------------- remaining baselines
EXTRA = [
    ("Self-Certainty", "Self-certainty"),
    ("Verbalized confidence", "Verbalized confidence"),
    ("Mean log-probability", "Mean log-prob"),
    ("DeepConf-LGC", "DeepConf-LGC"),
    ("EigV@8", "EigV@8"),
    ("Deg@8", "Deg@8"),
    ("Semantic entropy@8 (unweighted)", "Semantic entropy@8 (unweighted)"),
    ("Supervised prior signals", "Supervised prior signals"),
]


def app_extra_baselines(cells):
    ms = [("qwen8b", "Qwen3"), ("llama8b", "Llama"), ("ministral8b", "Ministral"), ("gptoss20b", "gpt-oss")]
    lines = []
    for lab, n in EXTRA:
        v = [macro(cells, n, auc, filt=lambda t, m=m: f"_{m}_" in t) for m, _ in ms] + [macro(cells, n, auc)]
        lines.append(lab + " & " + " & ".join(f3(x) for x in v) + r" \\")
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{Baselines not shown in Table~\ref{tab:main} (macro AUROC per backbone and over all 30 settings).}
\label{tab:app-extra}
\setlength{\tabcolsep}{2.6pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Qwen3} & \textbf{Llama} & \textbf{Ministral} & \textbf{gpt-oss} & \textbf{All} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    )
    (TAB / "app_extra.tex").write_text(tex)


# ----------------------------------------------------------------------------- label-free versus target labels
def table_few(cells=None):
    F = report("few_labels")
    g = lambda r, k: f3(r[k]) if r.get(k) is not None and np.isfinite(r[k]) else "--"
    lines = [f"{'All' if n == 'all' else n} & {g(F[n], 'in_lr')} & {g(F[n], 'in')} \\\\" for n in ["50", "100", "250", "all"]]
    lines += [r"\midrule", f"Label-free (0) & {g(F['50'], 'lf_lr')} & {g(F['50'], 'lf')} \\\\"]
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{Without target labels, \m\ ranks about as well as when trained on 250 of them. Macro AUROC over the """
        + str(F["n_settings"])
        + r""" settings with at least 500 responses, on the same 200 test responses per setting (five draws); LR: logistic regression}
\label{tab:few}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}
\toprule
\textbf{Target labels $n$} & \textbf{LR} & \textbf{Stacked} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    (TAB / "few.tex").write_text(tex)


def table_new_backbone(cells=None):
    """A larger new backbone (trac.analysis.transfer.new_backbone): Qwen3-32B scored by classifiers fitted
    on the four backbones of the main matrix (LOMDO, rank features), against training-free scores and
    the transferred views, with the in-domain classifier for reference."""
    R = report(f"new_backbone_{NEW_BACKBONE}")
    per = R["per"]

    def row(lab, key):
        if not all(key in per[d] for d in per):
            return None
        return lab + " & " + f3(np.mean([per[d][key] for d in per])) + r" \\"

    groups = [
        (r"\textit{Training-free}", [("Output length", "Output length"), ("Reward model", "Reward model")]),
        (
            r"\textit{Transferred from the four backbones}",
            [
                ("Token probs.", "Token probs. (LOMDO)"),
                (r"\quad $+$ early-exit signals", "Token probs. + early-exit signals (LOMDO)"),
                (r"\textbf{\m}", "TrAC (LOMDO)"),
            ],
        ),
        (r"\textit{In-domain (reference)}", [(r"\m", "TrAC (in-domain)")]),
    ]
    lines = []
    for title, rows in groups:
        body = [r for r in (row(lab, k) for lab, k in rows) if r]
        if body:
            lines += ([r"\midrule"] if lines else []) + [r"\multicolumn{2}{l}{" + title + r"} \\"] + body
    tex = (
        r"""\begin{table}[t]
\centering
\small
\caption{A larger new backbone: Qwen3-32B with thinking, scored without any of its labels by classifiers fitted on the four backbones of the main matrix for the other datasets (LOMDO, rank features). Macro AUROC over its """
        + str(R["n"])
        + r""" settings.}
\label{tab:new-backbone}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lc}
\toprule
& \textbf{AUROC} \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    (TAB / "new_backbone.tex").write_text(tex)
