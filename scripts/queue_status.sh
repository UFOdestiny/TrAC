#!/usr/bin/env bash
# Read-only report on running queue workers: jobs, GPUs, per-queue progress, latest errors, disk.
#   bash scripts/queue_status.sh [queue files, default configs/queues/[0-9]_*.txt]
source "$(dirname "$0")/env.sh"
R="$EXP_ROOT"
QUEUES=("${@:-configs/queues/[0-9]_*.txt}")
# shellcheck disable=SC2206
QUEUES=(${QUEUES[@]})
jobs_of() { grep -v '^#' "$1" | awk -F'|' 'NF{print ($8!="" ? $8 : ($5!="" ? $5 : $1))}'; }
echo "== $(date '+%F %T')"
echo "-- workers / active stages"
ps -eo pid,etime,args | grep -E "python -m trac\.pipeline\." | grep -v grep | cut -c1-150
echo "-- GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null
echo "-- orphan vLLM engines (ppid 1)"
ps -eo pid,ppid,etime,args | awk '$2==1 && /VLLM::EngineCore/' || true
echo "-- progress (done / failed / total)"
for f in "${QUEUES[@]}"; do
  d=0; x=0; tot=0
  for n in $(jobs_of "$f"); do
    tot=$((tot+1)); [ -f "$R/locks/$n/done" ] && d=$((d+1)); [ -f "$R/locks/$n/failed" ] && x=$((x+1))
  done
  echo "  $(basename "$f" .txt): $d done / $x failed / $tot"
done
echo "-- latest worker events"
for lg in "$R"/logs/queue_*_gpu*.log; do
  [ -f "$lg" ] && echo "  $(basename "$lg"): $(grep -E '\] (START|DONE|FAILED|DEFER)' "$lg" | tail -1)"
done
echo "-- errors in job logs (last 3)"
grep -l -a -E "Traceback|CUDA out of memory|Error:" "$R"/logs/*.log 2>/dev/null | tail -3
echo "-- disk"
df -h "$R" | tail -1
