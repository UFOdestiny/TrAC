#!/usr/bin/env bash
# Launch one detached queue worker per GPU on the shared job queue (resumable).
#   bash scripts/run_queue.sh main[,new_backbone] [gpus, default 0]
# Comma-separated matrices run in order on every worker (rewind.pipeline.queue).
# Logs: $EXP_ROOT/logs/queue_<first matrix>_gpu<i>.log
set -eu
source "$(dirname "$0")/env.sh"
MATRICES="${1:?usage: run_queue.sh <matrices: main, new_backbone, comma-separated> [gpus, default 0]}"
GPUS="${2:-0}"
NAME="${MATRICES%%,*}"
for g in ${GPUS//,/ }; do
  CUDA_VISIBLE_DEVICES="$g" nohup "$PY" -m rewind.pipeline.queue "$MATRICES" > "$EXP_ROOT/logs/queue_${NAME}_gpu${g}.log" 2>&1 &
  echo "worker gpu$g pid $! -> $EXP_ROOT/logs/queue_${NAME}_gpu${g}.log"
  sleep "${QUEUE_START_DELAY:-30}"   # stagger engine start-up
done
