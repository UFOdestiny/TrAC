#!/usr/bin/env bash
# Launch one detached queue worker per GPU on shared queue files (resumable).
#   bash scripts/run_queue.sh configs/queues/1_baselines.txt,configs/queues/2_trac.txt,configs/queues/3_controls.txt 0,1
# Comma-separated queues run in order on every worker. Logs: $EXP_ROOT/logs/queue_<first queue>_gpu<i>.log
set -eu
source "$(dirname "$0")/env.sh"
QUEUE="${1:?usage: run_queue.sh <queue files, comma-separated> [gpus, default 0,1]}"
GPUS="${2:-0,1}"
NAME="$(basename "${QUEUE%%,*}" .txt)"
for g in ${GPUS//,/ }; do
  CUDA_VISIBLE_DEVICES="$g" nohup "$PY" -m trac.pipeline.queue "$QUEUE" > "$EXP_ROOT/logs/queue_${NAME}_gpu${g}.log" 2>&1 &
  echo "worker gpu$g pid $! -> $EXP_ROOT/logs/queue_${NAME}_gpu${g}.log"
  sleep 30   # stagger engine start-up
done
