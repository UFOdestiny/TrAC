#!/usr/bin/env bash
# Run one GPU stage over several cache tags, sequentially on one GPU (finished tags are skipped).
#   bash scripts/run_stage.sh <stage> <gpu> <tag> [<tag> ...]
#   STAGE_ARGS="--mode trajectory" bash scripts/run_stage.sh latency 0 math500_qwen8b_k8
# stage: trajectory | sample_trajectory | prefix_consistency | hidden_states | nli | latency | endpoint
# Extra stage arguments come from $STAGE_ARGS. Logs: $EXP_ROOT/logs/<stage>_<tag>.log
set -u
source "$(dirname "$0")/env.sh"
STAGE="${1:?usage: run_stage.sh <stage> <gpu> <tag>...}"; GPU="${2:?gpu}"; shift 2
for t in "$@"; do
  log="$EXP_ROOT/logs/${STAGE}_${t}.log"
  echo "[run] $STAGE $t on GPU $GPU $(date +%T)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m "trac.pipeline.$STAGE" --gen-tag "$t" ${STAGE_ARGS:-} > "$log" 2>&1 \
    || echo "[run] FAILED $t (see $log)"
  tail -n 1 "$log"
done
echo "[run] done GPU $GPU $(date +%T)"
