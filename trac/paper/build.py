"""Build every table and figure of the paper from cached results (CPU).

  python -m trac.paper.build                    all tables + figures, and EXP_ROOT/paper_summary.json
  python -m trac.paper.build --only table_main,figs

Inputs: EXP_ROOT/{scores, features, feature_cache, auxiliary, trajectory, latency},
controls.json and prefix_consistency.json. Outputs: PAPER_DIR/table/*.tex and
PAPER_DIR/figures/{risk_coverage,pareto_quality,trajectory,reliability}.pdf.
Run single-threaded (OMP_NUM_THREADS=1).
"""
from __future__ import annotations

import argparse
import json

from trac import config as C
from trac.metrics import aurc, risk_at_coverage
from trac.paper import figures as F
from trac.paper import tables as T
from trac.paper.common import FIG, TAB, auc, latency_rel, load_cells, macro


def figs(cells):
    rows, _ = T.table_efficiency(cells)
    F.figures(cells, rows)
    return F.fig_trajectory(cells)


STEPS = {f.__name__: f for f in [
    T.table_main, T.table_ablation, T.table_decmatrix, T.table_robustness, T.table_transfer, T.table_label_eff,
    T.table_efficiency, T.app_matrix, T.app_blindspot, T.app_fusion, T.app_leakage_length, T.app_taxonomy,
    T.app_dataset, T.app_efficiency, T.app_calibration, F.figures, F.fig_trajectory, figs]}

SUMMARY_METHODS = ["TrAC", "TUP", "PCE", "SC@8 (primary)", "CISC@8", "TrAC + CF@8", "TrAC + HS", "Liu et al. (2024)",
                   "HS-probe", "P(True)", "SelfDoubt", "Supervised Consensus@8", "Abl: agreement only",
                   "Semantic entropy@8 (lik.)"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help=f"comma list of steps: {', '.join(STEPS)}")
    args = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    if args.only:
        for fn in args.only.split(","):
            print(fn, STEPS[fn](cells))
        return
    print(f"{len(cells)} cells ({sum(c['ok'] for c in cells.values())} in aggregates)")
    summary = {}
    T.table_main(cells); print("main")
    T.table_ablation(cells); print("ablation")
    T.table_decmatrix(); T.table_robustness(cells); print("controls")
    summary["transfer"] = T.table_transfer(cells); print("transfer")
    summary["label_eff"] = T.table_label_eff(cells); print("label efficiency")
    eff_rows, _ = T.table_efficiency(cells); print("efficiency")
    F.figures(cells, eff_rows); F.fig_trajectory(cells); print("figures")
    T.app_matrix(cells); summary["blindspot"] = {str(k): v for k, v in T.app_blindspot(cells).items()}
    T.app_fusion(cells); T.app_leakage_length(cells)
    summary["taxonomy"] = {k: v.tolist() for k, v in T.app_taxonomy(cells).items()}
    T.app_dataset(cells); print("appendix")
    summary["macro"] = {n: {"auroc": macro(cells, n, auc), "aurc": macro(cells, n, aurc),
                            "r10": macro(cells, n, lambda s, y: risk_at_coverage(s, y, 0.1))} for n in SUMMARY_METHODS}
    summary["latency"] = latency_rel()
    json.dump(summary, open(C.EXP_ROOT / "paper_summary.json", "w"), indent=1, default=float)
    print("saved", C.EXP_ROOT / "paper_summary.json")


if __name__ == "__main__":
    main()
