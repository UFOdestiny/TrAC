#!/usr/bin/env bash
# CPU analysis from cached GPU outputs to the paper's tables and figures.
#   bash scripts/run_analysis.sh [jobs]          (default: 6 parallel processes)
# Every step is single-threaded per process (sklearn with several OpenMP threads is much
# slower on a shared node) and parallelized over cells.
set -eu
source "$(dirname "$0")/env.sh"
JOBS="${1:-6}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MAIN=$(ls "$EXP_ROOT"/endpoint/*_k8.json.gz | xargs -n1 basename | sed 's/.json.gz$//')
step() { echo "[analysis] $* ($(date +%T))"; }

step "features (every endpoint cache)"
ls "$EXP_ROOT"/endpoint/*.json.gz | xargs -n1 basename | sed 's/.json.gz$//' \
  | xargs -P "$JOBS" -I{} "$PY" -m trac.analysis.features --tags {}
step "auxiliary statistics"
echo "$MAIN" | xargs -P "$JOBS" -I{} "$PY" -m trac.analysis.auxiliary --tags {}
step "out-of-fold scores"
echo "$MAIN" | xargs -P "$JOBS" -I{} sh -c "\"$PY\" -m trac.analysis.scores --tag {} > \"$EXP_ROOT/logs/scores_{}.log\" 2>&1"
"$PY" -m trac.analysis.scores --aggregate
step "controls, prefix consistency, within-fold AUROC, text trajectory"
"$PY" -m trac.analysis.controls
"$PY" -m trac.analysis.pc_comparison
echo "$MAIN" | xargs -P "$JOBS" -I{} "$PY" -m trac.analysis.within_fold --tag {}
"$PY" -m trac.analysis.within_fold --aggregate
echo "$MAIN" | xargs -P "$JOBS" -I{} "$PY" -m trac.analysis.text_trajectory --tag {}
"$PY" -m trac.analysis.text_trajectory --report
step "paper tables and figures -> ${PAPER_DIR:-latex}"
"$PY" -m trac.paper.build
step done
