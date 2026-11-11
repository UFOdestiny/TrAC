#!/usr/bin/env bash
# Run one GPU stage over cache tags, outside the queue. STAGE_ARGS supplies additional arguments.
#   bash scripts/run_stage.sh <stage> <gpu> <tag> [<tag> ...]
# The stage is looked up in rewind.pipeline, then in the optional companion packages.
set -euo pipefail
source "$(dirname "$0")/env.sh"
STAGE="${1:?usage: run_stage.sh <stage> <gpu> <tag>...}"
GPU="${2:?gpu}"
shift 2
if [ "$#" -eq 0 ]; then echo "At least one cache tag is required" >&2; exit 2; fi
MODULE="$("$PY" -m rewind.pipeline.queue --module "$STAGE")" || exit 2
FAILED=0
for tag in "$@"; do
  log="$EXP_ROOT/logs/${STAGE}_${tag}.log"
  echo "[run] $STAGE $tag on GPU $GPU"
  # shellcheck disable=SC2086
  if ! CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m "$MODULE" --gen-tag "$tag" ${STAGE_ARGS:-} > "$log" 2>&1; then
    echo "[run] FAILED $tag (see $log)" >&2
    FAILED=1
  fi
  tail -n 1 "$log"
done
exit "$FAILED"
