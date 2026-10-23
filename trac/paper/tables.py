"""LaTeX tables of the paper (written to PAPER_DIR/table, plus the reliability diagram)."""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.heads import clean, fit_fixed, oof, stack_fit_predict
from trac.analysis.methods import METHODS
from trac.metrics import aurc, brier, ece, risk_at_coverage
from trac.paper.common import (
    COL,
    DSETS,
    FIG,
    MATH,
    MODELS,
    TAB,
    _plt,
    auc,
    best_marks,
    boot_ci,
    f3,
    latency_rel,
    macro,
    parts,
    reps,
    tag_of,
)

# ----------------------------------------------------------------------------- main table
MAIN_ROWS = [
    ("Single", [("Mean log-probability", "Mean log-prob", "1.00$\\times$"),
                ("Self-certainty", "Self-certainty", "1.00$\\times$"),
                ("DeepConf-bottom", "DeepConf-bottom", "1.00$\\times$"),
                ("P(True)", "P(True)", "$+1$ fwd"),
                ("Verbalized confidence", "Verbalized confidence", "$+1$ dec"),
                ("SelfDoubt", "SelfDoubt", "$+1$ dec"),
                ("Supervised prior signals", "Supervised prior signals", "$+1$ fwd/dec"),
                ("Prefix consistency", "Prefix consistency", "PCCOST"),
                ("\\tupm\\ (passive)", "TUP", "1.00$\\times$"),
                ("\\pcem\\ (active)", "PCE", "1.00$\\times$"),
                ("\\textbf{\\m-0} (training-free)", "TrAC-0", "1.00$\\times$"),
                ("\\textbf{\\m}", "TrAC", "1.00$\\times$")]),
    ("White", [("HS-probe", "HS-probe", "1.00$\\times$"),
               ("\\citet{liu2024supervised}", "Liu et al. (2024)", "1.00$\\times$"),
               ("\\textbf{\\m\\ (+HS)}", "TrAC + HS", "1.00$\\times$")]),
    ("8-samp.", [("SelfCheckGPT@8", "SelfCheckGPT@8 (NLI)", None),
                 ("KLE@8", "KLE@8", None),
                 ("EigV@8", "EigV@8", None),
                 ("Deg@8", "Deg@8", None),
                 ("Semantic entropy@8", "Semantic entropy@8 (lik.)", None),
                 ("Supervised Consensus@8", "Supervised Consensus@8", None),
                 ("CISC@8", "CISC@8", None),
                 ("SC@8", "SC@8 (primary)", None),
                 ("\\textbf{\\m\\ (+CF@8)}", "TrAC + CF@8", None)]),
]


PROBED = {"PCE", "TrAC-0"}       # rows that pay for the re-elicitation queries


OURS = {"TrAC-0", "TrAC", "TrAC + HS", "TrAC + CF@8"}


def table_main(cells):
    lat = latency_rel()
    sc8 = f"{lat['sc_8']:.2f}$\\times$"
    trac_lat = f"{lat['trac']:.2f}$\\times$"
    trac0_lat = f"{lat['trac0']:.2f}$\\times$"
    learned = set(METHODS)                                    # probabilistic (learned) methods
    lines = []
    for grp, rows in MAIN_ROWS:
        names = [r[1] for r in rows]
        per_ds = {d: best_marks([macro(cells, n, auc, filt=lambda t, d=d: parts(t)[0] == d) for n in names])
                  for d, _ in DSETS}
        cols = {"auroc": best_marks([macro(cells, n, auc) for n in names]),
                "aurc": best_marks([macro(cells, n, aurc) for n in names], higher=False),
                "ece": best_marks([macro(cells, n, ece) if n in learned else np.nan for n in names], higher=False),
                "brier": best_marks([macro(cells, n, brier) if n in learned else np.nan for n in names], higher=False),
                "r10": best_marks([macro(cells, n, lambda s, y: risk_at_coverage(s, y, 0.1)) for n in names], higher=False)}
        lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{\\rotatebox{{90}}{{\\textit{{{grp}}}}}}}")
        for i, (label, name, cost) in enumerate(rows):
            cost = cost or sc8
            if cost == "PCCOST":
                pe = C.EXP_ROOT / "prefix_consistency.json"
                r = json.load(open(pe))["cost"].get("regen_tokens_over_original") if pe.exists() else None
                cost = f"$+{100 * r:.0f}\\%$ tok" if r else "--"
            if (name in OURS or name in PROBED) and grp != "8-samp.":
                cost = trac0_lat if name == "TrAC-0" else trac_lat
            if name in OURS and grp == "8-samp.":
                cost = f"{lat['sc_8'] + lat['trac'] - 1:.2f}$\\times$"
            row = " & ".join([per_ds[d][i] for d, _ in DSETS])
            pre = "\\rowcolor{gray!15}\\cellcolor{white}" if name in OURS else ""
            lines.append(f"{pre} & {label} & {cost} & {row} & {cols['auroc'][i]} & {cols['aurc'][i]} & "
                         f"{cols['ece'][i]} & {cols['brier'][i]} & {cols['r10'][i]} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    head = " & ".join(n for _, n in DSETS)
    tex = r"""\begin{table*}[t]
\centering
\small
\caption{Overall performance. Dataset columns report AUROC averaged over backbones; Macro averages all
settings. ECE and Brier apply to methods that output probabilities; Risk@10\% is the error rate among the 10\%
most-confident responses. Best per budget block in \textbf{bold}, second \underline{underlined}. Cost is the
measured wall-clock relative to one original generation (median over audits, Section~\ref{sec:rq5}).}
\setlength{\tabcolsep}{2.6pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llc cccccccc ccccc}
\toprule
& \multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Cost}}
& \multicolumn{8}{c}{\textbf{AUROC}$\uparrow$}
& \multicolumn{5}{c}{\textbf{Macro}} \\
\cmidrule(lr){4-11}\cmidrule(lr){12-16}
& & & """ + head + r""" & AUROC$\uparrow$ & AURC$\downarrow$ & ECE$\downarrow$ & Brier$\downarrow$ & Risk@10\%$\downarrow$ \\
\midrule
""" + "\n".join(lines) + r"""
\end{tabular}}
\label{tab:main}
\end{table*}
"""
    (TAB / "result.tex").write_text(tex)


# ----------------------------------------------------------------------------- ablation
ABL = [("(a) Component contribution", [("\\tupm\\ only", "TUP"), ("\\pcem\\ only", "PCE"),
                                       ("\\pcem\\ $+$ \\tupm, logistic head", "Abl: logistic head"),
                                       ("\\pcem\\ $+$ \\tupm\\ ($=$ \\m)", "TrAC")]),
       ("(b) Re-elicitation along the trace", [("Completed prefix only", "PCE (completed prefix only)"),
                                              ("Trajectory only", "Trajectory only"),
                                              ("Both ($=$ \\pcem)", "PCE"),
                                              ("\\m\\ without trajectory", "TrAC without trajectory"),
                                              ("\\m\\ (with trajectory)", "TrAC")])]


TRACECOND = [("Question-only re-elicitation", "Abl: question-only probe"),
                                            ("Shuffled-trace re-elicitation", "Abl: shuffled-trace probe"),
                                            ("Teacher-forced likelihood", "Abl: teacher-forced support"),
                                            ("Re-elicitation on its own completed trace", "Abl: completed-trace probe")]


SELFVER = [("P(True)", "P(True)"), ("Completed-prefix \\pcem", "PCE (completed prefix only)"),
           ("P(True) $+$ completed-prefix \\pcem", "Abl: P(True) + PCE")]


ABL_ENDPOINT = [("Agreement only", "Abl: agreement only"), ("Likelihood only", "Abl: likelihood only"),
                ("Agreement $+$ likelihood", "Abl: completed-trace probe"),
                ("$+$ re-elicitation stability", "Abl: + re-elicitation distribution"),
                ("$+$ likelihood gap ($=$ completed-prefix \\pcem)", "PCE (completed prefix only)")]


def _abl_lines(cells, groups):
    lines = []
    for title, rows in groups:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        vals = [macro(cells, n, auc) for _, n in rows]
        for (label, n), v in zip(rows, vals):
            lo, hi = boot_ci(cells, n)
            b = v == max(vals)
            fmt = (lambda s: r"\textbf{" + s + "}") if b else (lambda s: s)
            lines.append(f"{label} & {fmt(f3(v))} & {fmt('[' + f3(lo) + ', ' + f3(hi) + ']')} & "
                         f"{fmt(f3(macro(cells, n, aurc)))} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    return lines


def table_ablation(cells):
    lines = _abl_lines(cells, ABL)
    ep = _abl_lines(cells, [("Trace-conditioning controls", TRACECOND), ("Completed-prefix features", ABL_ENDPOINT),
                            ("Comparison with self-verification", SELFVER)])
    (TAB / "app_ablation_endpoint.tex").write_text(r"""\begin{table}[t]
\centering
\small
\caption{Additional ablations: trace-conditioning controls, decomposition of the completed-prefix features of \pcem,
and comparison with self-verification (macro AUROC with 95\% hierarchical-bootstrap intervals, same stacked head and
folds as Table~\ref{tab:ablation}).}
\label{tab:app-ablation-endpoint}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Representation} & \textbf{AUROC $\uparrow$} & \textbf{95\% CI} & \textbf{AURC $\downarrow$} \\
\midrule
""" + "\n".join(ep) + r"""
\end{tabular}}
\end{table}
""")
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Macro ablation results. All learned rows except the logistic-head row use the same stacked head and folds; 95\% confidence
intervals come from a hierarchical bootstrap over settings with 2,000 resamples.}
\label{tab:ablation}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Representation} & \textbf{AUROC $\uparrow$} & \textbf{95\% CI} & \textbf{AURC $\downarrow$} \\
\midrule
""" + "\n".join(lines) + r"""
\end{tabular}}
\end{table}
"""
    (TAB / "ablation.tex").write_text(tex)


# ----------------------------------------------------------------------------- decoding matrix / seeds / cues
def table_decmatrix(cells=None):
    ctrl = json.load(open(C.EXP_ROOT / "controls.json"))
    conds = ["T0.7 -> greedy (default)", "greedy -> greedy", "T0.2 -> greedy", "T1.0 -> greedy", "T0.7 -> T0.7"]
    names = {conds[0]: "$T{=}0.7\\to$ greedy (default)", conds[1]: "greedy $\\to$ greedy",
             conds[2]: "$T{=}0.2\\to$ greedy", conds[3]: "$T{=}1.0\\to$ greedy", conds[4]: "$T{=}0.7\\to T{=}0.7$"}
    lines = []
    for d, dn in [("math500", "MATH500"), ("minerva", "Minerva")]:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{dn}, macro over backbones}}}} \\\\")
        for c in conds:
            rs = [ctrl["decoding"].get(f"{d}/{m}/{c}") for m, _ in MODELS]
            rs = [r for r in rs if r]
            acc, agr = np.mean([r["acc"] for r in rs]), np.mean([r["agree"] for r in rs])
            pce, tr = np.mean([r["Endpoint"] for r in rs]), np.mean([r["Endpoint + TUP"] for r in rs])
            t0 = [r.get("TrAC-0") for r in rs]
            t0 = f3(np.mean(t0)) if all(v is not None and np.isfinite(v) for v in t0) else "--"
            lines.append(f"{names[c]} & {f3(acc)} & {f3(agr)} & {f3(pce)} & {f3(tr)} & {t0} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Decoding ablation, varying the temperature of the original generation and of re-elicitation (MATH500
and Minerva, averaged over the three backbones). Agr.\ is the rate at which the re-elicited answer matches the
final answer. Each condition re-elicits with the default answer prompt only, so the Re-elic.\ column uses the
agreement, likelihood, and confidence of that answer, and the next column adds \tupm. The last column is the
training-free \m-0 score, whose trajectory probes are decoded as in the main setting.}
\label{tab:decmatrix}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Generation $\to$ re-elic.} & \textbf{Acc} & \textbf{Agr.} & \textbf{Re-elic.} & \textbf{Re-elic.$+$\tupm} & \textbf{\m-0} \\
\midrule
""" + "\n".join(lines) + r"""
\end{tabular}}
\end{table}
"""
    (TAB / "decmatrix.tex").write_text(tex)


def table_robustness(cells):
    """generation seeds + cue variants."""
    ctrl = json.load(open(C.EXP_ROOT / "controls.json"))
    lines = []
    for m, mn in MODELS:
        for d, dn in [("math500", "MATH500"), ("olympiad", "Olympiad")]:
            a = ctrl["seeds"][f"{d}/{m}"]
            lines.append(f"{dn} & {mn} & ${f3(a['acc'][0])}\\pm{f3(a['acc'][1])}$ & "
                         f"${f3(a['Endpoint'][0])}\\pm{f3(a['Endpoint'][1])}$ & "
                         f"${f3(a['Endpoint + TUP'][0])}\\pm{f3(a['Endpoint + TUP'][1])}$ & "
                         f"${f3(a['TrAC-0'][0])}\\pm{f3(a['TrAC-0'][1])}$ \\\\")
    cue_auc = defaultdict(list)
    for t, c in cells.items():
        if not c["ok"]:
            continue
        B, y = c["B"], c["y"]
        for k, X in enumerate([np.hstack([B["AGREE"], B["FLL"][:, :1]]), B["CUES"][:, 0:2], B["CUES"][:, 2:4]]):
            s = oof(X, y)
            if s is not None:
                cue_auc[k].append(roc_auc_score(y, s))
    cue = [np.mean(cue_auc[k]) for k in range(3)]
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Robustness. Top: AUROC over three independent generation seeds (mean $\pm$ std; the Re-elic.\ column
uses the default answer prompt only; \m-0 is the training-free trajectory score). Bottom: macro AUROC of a logistic head on agreement
and answer likelihood elicited with three answer prompts (the default and two paraphrases).}
\label{tab:seeds}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{Acc} & \textbf{Re-elic.} & \textbf{Re-elic.$+$\tupm} & \textbf{\m-0} \\
\midrule
""" + "\n".join(lines) + r"""
\midrule
\multicolumn{5}{l}{\textit{Answer prompt (agreement $+$ likelihood)}} & \textbf{AUROC} \\
\multicolumn{5}{l}{\texttt{The final answer is}} & """ + f3(cue[0]) + r""" \\
\multicolumn{5}{l}{\texttt{Therefore, the answer is}} & """ + f3(cue[1]) + r""" \\
\multicolumn{5}{l}{\texttt{So the final result is}} & """ + f3(cue[2]) + r""" \\
\bottomrule
\end{tabular}}
\end{table}
"""
    (TAB / "robustness.tex").write_text(tex)


def table_transfer(cells):
    ok = {t: c for t, c in cells.items() if c["ok"]}
    R = {t: reps(c["B"]) for t, c in ok.items()}
    ind = {t: {"TUP": auc(c["s"]["TUP"], c["y"]), "PCE": auc(c["s"]["PCE"], c["y"]),
               "TrAC": auc(c["s"]["TrAC"], c["y"])} for t, c in ok.items()}

    def run(test_of):            # test_of(t) -> group key; train on cells of other groups
        res = defaultdict(dict)
        for g in sorted({test_of(t) for t in ok}):
            te = [t for t in ok if test_of(t) == g]
            tr = [t for t in ok if test_of(t) != g]
            for r in ["TUP", "PCE", "TrAC"]:
                Xtr = np.vstack([R[t][r] for t in tr]); ytr = np.concatenate([ok[t]["y"] for t in tr])
                for t in te:
                    res[t][r] = auc(stack_fit_predict(Xtr, ytr, R[t][r]), ok[t]["y"])
        return res

    lodo = run(lambda t: parts(t)[0])
    lomo = run(lambda t: parts(t)[1])
    glob_ = run(lambda t: t)                                      # leave-one-cell-out global head
    lines = []
    for d, dn in DSETS:
        ts = [t for t in ok if parts(t)[0] == d]
        if not ts:
            continue
        a, b = np.mean([lodo[t]["TrAC"] for t in ts]), np.mean([ind[t]["TrAC"] for t in ts])
        best = max(a, b)
        fm = lambda v: (r"\textbf{" + f3(v) + "}") if v == best else f3(v)
        lines.append(f"{dn} & {fm(a)} & {fm(b)} & ${a - b:+.3f}$ \\\\")
    agg = {r: [np.mean([x[t][r] for t in ok]) for x in (lodo, glob_, lomo)] + [np.mean([ind[t][r] for t in ok])]
           for r in ["TUP", "PCE", "TrAC"]}
    bot = []
    for r, lab in [("TUP", "\\quad \\tupm"), ("PCE", "\\quad \\pcem"), ("TrAC", "\\quad \\m")]:
        v = agg[r]
        bot.append(f"{lab} & {f3(v[0])} & {f3(v[1])} & {f3(v[2])} & {f3(v[3])} \\\\")
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Cross-dataset and cross-model transfer of the \m head. Top: per-target leave-one-dataset-out (LODO)
with the in-domain score for reference (macro over backbones). Bottom: the three representations under LODO,
a single global head (leave-one-setting-out), leave-one-model-out (LOMO), and in-domain fitting.}
\label{tab:transfer}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
\textbf{Held-out target} & \textbf{Transfer} & \textbf{In-domain} & \textbf{$\Delta$} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{lcccc}
\toprule
& \textbf{LODO} & \textbf{Global} & \textbf{LOMO} & \textbf{In-domain} \\
\midrule
""" + "\n".join(bot) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "transfer.tex").write_text(tex)
    return agg


def table_label_eff(cells):
    """Fixed 40% test split; label fractions of the remaining 60%. All fractions use the same
    cells (those whose smallest labeled subset still holds 3 examples per class in every repeat)
    and a logistic head, so rows are directly comparable."""
    from sklearn.model_selection import StratifiedShuffleSplit
    fr = [0.05, 0.10, 0.25, 0.50, 1.0]
    plan = {}
    for t, c in cells.items():
        if not c["ok"]:
            continue
        y = c["y"]; ok = True; sp = []
        for rep in range(3):
            tr0, te = next(StratifiedShuffleSplit(1, test_size=0.4, random_state=rep).split(y.reshape(-1, 1), y))
            perm = np.random.RandomState(rep).permutation(tr0)
            idx = {f: (perm[:int(round(f * len(tr0)))] if f < 1 else tr0) for f in fr}
            if min(y[idx[fr[0]]].sum(), (1 - y[idx[fr[0]]]).sum()) < 3:
                ok = False; break
            sp.append((idx, te))
        if ok:
            plan[t] = sp
    res = defaultdict(lambda: defaultdict(list))
    for t, sp in plan.items():
        R, y = reps(cells[t]["B"]), cells[t]["y"]
        for idx, te in sp:
            for f in fr:
                for r in ["PCE", "TrAC"]:
                    X = clean(R[r])
                    m = fit_fixed("lr", X[idx[f]], y[idx[f]])
                    res[r][f].append(auc(m.predict_proba(X[te])[:, 1], y[te]))
    full = np.mean(res["TrAC"][1.0])
    lines = [f"{f * 100:g}\\% & {f3(np.mean(res['PCE'][f]))} & {f3(np.mean(res['TrAC'][f]))} & "
             f"{100 * np.mean(res['TrAC'][f]) / full:.1f}\\% \\\\" for f in fr]
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Label efficiency on a fixed 40\% test split (three repeats). Only the labeled fraction of the
remaining 60\% changes. All rows use the same """ + str(len(plan)) + r""" settings and a logistic head;
retention is relative to \m\ with all labels.}
\label{tab:label-efficiency}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
\textbf{Labels} & \textbf{\pcem} & \textbf{\m} & \textbf{Retention} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "label_efficiency.tex").write_text(tex)
    return {r: {f: float(np.mean(res[r][f])) for f in fr} for r in res}, len(plan)


# ----------------------------------------------------------------------------- efficiency
def table_efficiency(cells):
    rel = {os.path.basename(f)[len("endpoint_"):-5]: json.load(open(f))["relative"]
           for f in glob.glob(str(C.LATENCY / "endpoint_*.json"))}
    lat = latency_rel()
    q = {"SC@2": "sc2", "SC@4": "sc4"}
    aurocs = {"SC@8": macro(cells, "SC@8 (primary)", auc), "CF": macro(cells, "TrAC + CF@8", auc)}
    for k, a in q.items():
        v = [auc(c["aux"][a], c["y"]) for c in cells.values() if c["ok"] and a in c["aux"]]
        aurocs[k] = float(np.mean(v)) if v else float("nan")
    m = lambda n: macro(cells, n, auc)
    rows = [("\\tupm\\ (no extra call)", 1.0, m("TUP")),
            ("\\m-0 (training-free)", lat["traj_20_4"], m("TrAC-0")),
            ("\\m, completed prefix only", lat["full_0_0"], m("TrAC without trajectory")),
            ("\\m, 4 checkpoints", lat["full_4_4"], m("Abl: trajectory, 4 checkpoints")),
            ("\\m, 8 checkpoints", lat["full_8_4"], m("Abl: trajectory, 8 checkpoints")),
            ("\\m, greedy answers only", lat["full_20_0"], m("Abl: trajectory, greedy only")),
            ("\\m\\ (20 checkpoints, $S{=}4$)", lat["full_20_4"], m("TrAC")),
            ("SC@2", lat["sc_2"], aurocs["SC@2"]),
            ("\\tupm\\ $+$ SC@2 (supervised)", lat["sc_2"], m("TUP + SC@2")),
            ("\\m\\ $+$ SC@2 (supervised)", lat["sc_2"] + lat["full_20_4"] - 1, m("TrAC + SC@2")),
            ("SC@4", lat["sc_4"], aurocs["SC@4"]),
            ("SC@8", lat["sc_8"], aurocs["SC@8"]),
            ("\\m\\ (+CF@8)", lat["sc_8"] + lat["full_20_4"] - 1, aurocs["CF"])]
    lines = [f"{n} & {l:.2f}$\\times$ & {f3(a)} \\\\" for n, l, a in rows]
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Measured wall-clock cost relative to one original generation (median over four audits on one H200,
64 queries in batches of 16) and macro AUROC.}
\label{tab:pareto-main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{Latency} & \textbf{AUROC} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "pareto.tex").write_text(tex)
    return rows, rel


# ----------------------------------------------------------------------------- appendix tables
def app_matrix(cells):
    single = ["SelfDoubt", "TUP", "PCE", "TrAC-0", "TrAC"]
    multi = ["SC@8 (primary)", "TrAC + CF@8"]
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
        lines.append(f"{first} & {mn} & & & " + " & ".join(best_marks(list(v[:len(single)])) + best_marks(list(v[len(single):]))) + " \\\\")
    lines.append("\\bottomrule")
    tex = r"""\begin{table*}[t]
\centering
\small
\caption{Complete AUROC results for every (dataset, backbone) setting, with macro averages per backbone over the
settings that enter the aggregates. Settings with fewer than ten correct or ten incorrect responses (AIME for
Llama-3.1-8B and Ministral-8B) are reported but excluded. Best values are marked separately among the single-trace
methods and among the eight-sample methods.}
\label{tab:app-matrix}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccccccc}
\toprule
\textbf{Dataset} & \textbf{Model} & $n$ & \textbf{Acc.} & \textbf{SelfDoubt} & \textbf{\tupm} & \textbf{\pcem} & \textbf{\m-0} & \textbf{\m} & \textbf{SC@8} & \textbf{\m\ (+CF@8)} \\\\
\midrule
""" + "\n".join(lines) + r"""
\end{tabular}
\end{table*}
"""
    (TAB / "app_matrix.tex").write_text(tex)


def app_blindspot(cells):
    lines = []
    out = {}
    for th in [0.625, 0.75, 0.875, 1.0]:
        n = w = 0
        v = defaultdict(list)
        for c in cells.values():
            if not c["ok"]:
                continue
            m = c["s"]["SC@8 (primary)"] >= th - 1e-9
            y = c["y"][m]
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            n += m.sum(); w += (1 - y).sum()
            for k, nm in [("SC", "SC@8 (primary)"), ("TUP", "TUP"), ("PCE", "PCE"),
                          ("TrAC", "TrAC"), ("HS", "HS-probe")]:
                s = c["s"][nm][m]
                v[k].append(auc(s, y) if np.ptp(s) > 0 else 0.5)
        r = {k: float(np.mean(x)) for k, x in v.items()}
        out[th] = (n, w, len(v["SC"]), r)
        lines.append(f"{th:.3f} & {len(v['SC'])} & {n:,} & {w:,} & {f3(r['SC'])} & {f3(r['TUP'])} & {f3(r['PCE'])} & "
                     f"{f3(r['TrAC'])} & {f3(r['HS'])} \\\\")
    tex = r"""\begin{table}[t]
\centering
\small
\caption{AUROC on increasingly high-agreement subsets (vote share of the final answer among eight samples $\geq$
threshold). Settings enter a row when the subset holds at least five correct and five incorrect responses; $n$ and
Wrong count the responses of those settings. HS is the white-box HS-probe.}
\label{tab:app-blindspot}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ccccccccc}
\toprule
\textbf{Vote $\geq$} & \textbf{Settings} & $n$ & \textbf{Wrong} & \textbf{SC} & \textbf{\tupm} & \textbf{\pcem} & \textbf{\m} & \textbf{HS} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    (TAB / "app_blindspot.tex").write_text(tex)
    return out


def app_fusion(cells):
    lines = []
    for d, dn in DSETS:
        for m, mn in MODELS:
            t = tag_of(d, m)
            if t not in cells or not cells[t]["ok"]:
                continue
            c = cells[t]
            v = [auc(c["s"][n], c["y"]) for n in ["SC@8 (primary)", "Semantic entropy@8 (lik.)", "CISC@8",
                                                   "TrAC + CF@8"]]
            lines.append(f"{dn} & {mn} & " + " & ".join(f3(x) for x in v) + f" & ${v[3] - v[0]:+.3f}$ \\\\")
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Matched-budget fusion (\cfm) against eight-sample baselines, per setting.}
\label{tab:app-fusion}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccccc}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{SC@8} & \textbf{SE@8} & \textbf{CISC@8} & \textbf{\cfm} & $\boldsymbol{\Delta}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
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
            for k, nm in [("PCE", "PCE"), ("TrAC", "TrAC")]:
                sub[lab][k].append(auc(c["s"][nm][msk], y))
            sub[lab]["frac"].append(msk.mean())
    allf = np.concatenate([c["aux"]["in_trace"] for t, c in cells.items()
                           if c["ok"] and parts(t)[0] in MATH and "in_trace" in c["aux"]])
    share = {"stated": allf.mean(), "not stated": 1 - allf.mean()}
    lines = [f"Answer {lab} in trace & {len(sub[lab]['TrAC'])} & {100 * share[lab]:.0f}\\% & "
             f"{f3(np.mean(sub[lab]['PCE']))} & \\textbf{{{f3(np.mean(sub[lab]['TrAC']))}}} \\\\"
             for lab in ["stated", "not stated"]]
    (TAB / "app_leakage.tex").write_text(r"""\begin{table}[t]
\centering
\small
\caption{Answer-leakage control on the mathematical settings. Traces are split by whether the reasoning body
already contains the final answer string; Share is the fraction of all mathematical responses in each split,
and a setting enters a row when that split holds at least five correct and five incorrect responses.}
\label{tab:app-leakage}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
\textbf{Subset} & \textbf{Settings} & \textbf{Share} & \textbf{\pcem} & \textbf{\m} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
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
            for k, nm in [("TUP", "TUP"), ("PCE", "PCE"), ("TrAC", "TrAC")]:
                ter[i][k].append(auc(c["s"][nm][msk], y))
    lines = [f"{lab} & {len(ter[i]['TrAC'])} & {f3(np.mean(ter[i]['TUP']))} & {f3(np.mean(ter[i]['PCE']))} & "
             f"\\textbf{{{f3(np.mean(ter[i]['TrAC']))}}} \\\\" for i, lab in enumerate(["Short", "Medium", "Long"])]
    (TAB / "app_length.tex").write_text(r"""\begin{table}[t]
\centering
\small
\caption{Length control. AUROC within per-setting output-length tertiles (macro over settings with at least five
correct and five incorrect responses in the tertile).}
\label{tab:app-length}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
\textbf{Length tertile} & \textbf{Settings} & \textbf{\tupm} & \textbf{\pcem} & \textbf{\m} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return sub, ter


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
        v = cnt[m]; s = v.sum()
        lines.append(f"{mn} & {int(s):,} & " + " & ".join(f"{100 * x / s:.1f}" for x in v) + " \\\\")
    (TAB / "app_taxonomy.tex").write_text(r"""\begin{table}[t]
\centering
\small
\caption{What re-elicitation returns when the original answer is wrong (\% of incorrect responses, all 24
settings): the correct answer, the same wrong answer, a different wrong answer, or no parsable answer.}
\label{tab:app-taxonomy}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Wrong} & \textbf{Correct} & \textbf{Same wrong} & \textbf{Diff.\ wrong} & \textbf{Unparsable} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    return cnt


def app_dataset(cells):
    lines = []
    reg = {"gsm8k": "Grade-school word problems", "math500": "Competition mathematics",
           "minerva": "Technical quantitative", "olympiad": "Olympiad-level problems",
           "aime3": "AIME 2024--2026", "gpqa_diamond": "Graduate-level science (MC)",
           "mmlupro": "Multitask knowledge (MC, 10-way)", "triviaqa": "Open-domain factual QA"}
    for d, dn in DSETS:
        ts = [t for t in cells if parts(t)[0] == d]
        n = len(cells[ts[0]]["y"])
        acc = ", ".join(f"{cells[tag_of(d, m)]['y'].mean():.2f}" for m, _ in MODELS if tag_of(d, m) in cells)
        lines.append(f"{dn} & {n} & {acc} & {reg[d]} \\\\")
    (TAB / "app_dataset.tex").write_text(r"""\begin{table}[t]
\centering
\small
\caption{Benchmarks. Acc.\ lists final-answer accuracy for Qwen3-8B, Llama-3.1-8B, and Ministral-8B.}
\label{tab:app-dataset}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccl}
\toprule
\textbf{Benchmark} & \textbf{Problems} & \textbf{Acc.} & \textbf{Regime} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def app_efficiency(cells):
    rows = []
    names = {"math500_qwen8b": "MATH500 / Qwen3-8B", "olympiad_qwen8b": "Olympiad / Qwen3-8B",
             "math500_llama8b": "MATH500 / Llama-3.1-8B", "math500_ministral8b": "MATH500 / Ministral-8B"}
    for f in sorted(glob.glob(str(C.LATENCY / "endpoint_*.json"))):
        k = os.path.basename(f)[len("endpoint_"):-5]
        j = json.load(open(f))
        r = j["relative"]
        ft = C.LATENCY / f"trajectory_{k}.json"
        t = json.load(open(ft))["relative"] if ft.exists() else {}
        tok = np.mean([b["gen_tokens"] for b in j["batches"]]) / j["batch"]
        rows.append(f"{names.get(k, k)} & {tok:,.0f} & {t.get('full_0_0', np.nan):.3f}$\\times$ & {t.get('traj_20_4', np.nan):.3f}$\\times$ & "
                    f"{t.get('full_20_4', np.nan):.3f}$\\times$ & "
                    f"{r['sc_2']:.2f}$\\times$ & {r['sc_4']:.2f}$\\times$ & {r['sc_8']:.2f}$\\times$ \\\\")
    (TAB / "app_efficiency.tex").write_text(r"""\begin{table}[t]
\centering
\small
\caption{Per-audit wall-clock cost relative to one original generation (one H200, 64 queries in batches of 16,
fresh generations for every method). Tokens is the mean length of the original trace. End only issues every
completed-prefix query of \pcem; \m-0 the greedy and sampled checkpoint answers; \m\ every query of its features.}
\label{tab:app-efficiency}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Audit} & \textbf{Tokens} & \textbf{End only} & \textbf{\m-0} & \textbf{\m} & \textbf{SC@2} & \textbf{SC@4} & \textbf{SC@8} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def app_calibration(cells):
    """In-domain ECE / Brier / NLL of the learned methods; ECE / Brier / NLL / AUROC of the TrAC head under
    leave-one-dataset-out and leave-one-model-out transfer; fixed-threshold abstention; reliability diagram."""
    ok = {t: c for t, c in cells.items() if c["ok"]}
    nll = lambda s, y: float(-np.mean(y * np.log(np.clip(s, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - s, 1e-6, 1))))
    learned = [("\\tupm", "TUP"), ("Supervised prior signals", "Supervised prior signals"),
               ("\\m", "TrAC"), ("HS-probe", "HS-probe"),
               ("\\citet{liu2024supervised}", "Liu et al. (2024)"), ("\\m\\ (+HS)", "TrAC + HS"),
               ("Supervised Consensus@8", "Supervised Consensus@8"), ("\\m\\ (+CF@8)", "TrAC + CF@8")]
    rows = []
    for lab, n in learned:
        v = [(ece(c["s"][n], c["y"]), brier(c["s"][n], c["y"]), nll(c["s"][n], c["y"])) for c in ok.values() if n in c["s"]]
        rows.append(f"{lab} & In-domain & " + " & ".join(f3(x) for x in np.mean(v, 0)) + " & -- \\\\")
    # transfer of the TrAC head
    R = {t: reps(c["B"])["TrAC"] for t, c in ok.items()}
    trans = {}
    for name, key in [("LODO", 0), ("LOMO", 1)]:
        pred = {}
        for g in sorted({parts(t)[key] for t in ok}):
            te = [t for t in ok if parts(t)[key] == g]; tr = [t for t in ok if parts(t)[key] != g]
            Xtr = np.vstack([R[t] for t in tr]); ytr = np.concatenate([ok[t]["y"] for t in tr])
            for t in te:
                pred[t] = stack_fit_predict(Xtr, ytr, R[t])
        trans[name] = pred
        v = [(ece(pred[t], ok[t]["y"]), brier(pred[t], ok[t]["y"]), nll(pred[t], ok[t]["y"]), auc(pred[t], ok[t]["y"])) for t in ok]
        rows.append(f"\\m & {name} & " + " & ".join(f3(x) for x in np.mean(v, 0)) + " \\\\")
    # fixed-threshold abstention: accept responses with score >= tau
    ab = []
    for tau in [0.5, 0.7, 0.9]:
        for name, get in [("In-domain", lambda t: ok[t]["s"]["TrAC"]), ("LODO", lambda t: trans["LODO"][t])]:
            cov, risk, gap = [], [], []
            for t in ok:
                s, y = get(t), ok[t]["y"]
                m = s >= tau
                cov.append(m.mean())
                if m.sum():
                    risk.append(1 - y[m].mean()); gap.append(1 - s[m].mean())
            ab.append(f"{tau:.1f} & {name} & {100 * np.mean(cov):.1f}\\% & {f3(np.mean(risk))} & {f3(np.mean(gap))} \\\\")
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Calibration. Top: macro ECE, Brier score, and negative log-likelihood of the learned methods in-domain,
and of the \m\ head under leave-one-dataset-out (LODO) and leave-one-model-out (LOMO) transfer (with AUROC).
Bottom: abstaining below a fixed threshold $\tau$ on the \m\ score: macro coverage, error rate among the
accepted responses, and the error rate the scores predict for them ($1-\bar u$).}
\label{tab:app-calibration}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Method} & \textbf{Fit} & \textbf{ECE} & \textbf{Brier} & \textbf{NLL} & \textbf{AUROC} \\\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}

\vspace{4pt}
\begin{tabular}{llccc}
\toprule
$\tau$ & \textbf{Fit} & \textbf{Coverage} & \textbf{Risk} & \textbf{Predicted} \\\\
\midrule
""" + "\n".join(ab) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "app_calibration.tex").write_text(tex)
    # reliability diagram (pooled over settings)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.7 / 2.54, 5.8 / 2.54))
    bins = np.linspace(0, 1, 11)
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.9)
    for lab, get, col, mk, ls in [("In-domain", lambda t: ok[t]["s"]["TrAC"], COL["TrAC"], "o", "-"),
                                  ("LODO", lambda t: trans["LODO"][t], COL["TUP"], "s", "--")]:
        s = np.concatenate([get(t) for t in ok]); y = np.concatenate([ok[t]["y"] for t in ok])
        idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
        xs = [s[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ax.plot(xs, ys, marker=mk, ls=ls, color=col, ms=3.5, label=lab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted correctness"); ax.set_ylabel("Observed accuracy")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "reliability.pdf"); plt.close(fig)
