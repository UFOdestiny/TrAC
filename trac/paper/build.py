"""Build the paper's tables and figures from the scores and reports under EXP_ROOT (CPU).

  python -m trac.paper.build                       every table and figure
  python -m trac.paper.build --only result,teaser  selected steps

Writes PAPER_DIR/table/*.tex and PAPER_DIR/figures/*.pdf; the framework figure compiles
PAPER_DIR/figures/framework.tex with XeLaTeX. Run single-threaded (OMP_NUM_THREADS=1).
"""

from __future__ import annotations

import argparse

from trac.paper import figures as F
from trac.paper import tables as T
from trac.paper.common import FIG, TAB, load_cells

STEPS = {  # step -> function; tables are named after the file they write
    "result": T.table_main,
    "ablation": T.table_ablation,
    "transfer": T.table_transfer,
    "transfer_free": T.table_transfer_free,
    "label_efficiency": T.table_label_eff,
    "few": T.table_few,
    "new_backbone": T.table_new_backbone,
    "cost": T.table_cost,
    "app_matrix": T.app_matrix,
    "app_extra": T.app_extra_baselines,
    "app_dataset": T.app_dataset,
    "app_calibration": T.app_calibration,
    "app_blindspot": T.app_blindspot,
    "app_fusion": T.app_fusion,
    "app_leakage_length": T.app_leakage_length,
    "app_taxonomy": T.app_taxonomy,
    "app_comparisons": T.app_comparisons,
    "framework": F.fig_framework,
    "teaser": F.fig_teaser,
    "insight": F.fig_insight,
    "scale": F.fig_scale,
    "trajectory": F.fig_trajectory,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help=f"comma-separated steps: {', '.join(STEPS)}")
    args = ap.parse_args()
    steps = args.only.split(",") if args.only else list(STEPS)
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown steps: {', '.join(unknown)}")
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    print(f"{len(cells)} settings ({sum(c['ok'] for c in cells.values())} in aggregates)")
    for step in steps:
        STEPS[step](cells)
        print("built", step, flush=True)


if __name__ == "__main__":
    main()
