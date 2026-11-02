#!/usr/bin/env bash
# Read-only report on queue workers: active stages, GPUs, per-queue progress, failed jobs, disk.
#   bash scripts/queue_status.sh [queue files, default configs/queues/main.txt and new_backbone.txt]
source "$(dirname "$0")/env.sh"
if [ "$#" -gt 0 ]; then
  QUEUES=("$@")
else
  QUEUES=(configs/queues/main.txt configs/queues/new_backbone.txt)
fi
echo "== $(date '+%F %T')"
echo "-- active stages"
ps -eo pid,etime,args | grep -E "python -m trac\.pipeline\." | grep -v grep | cut -c1-150
echo "-- GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null
echo "-- progress (done / failed / total jobs)"
for f in "${QUEUES[@]}"; do
  "$PY" - "$f" <<'PY'
import sys
from trac.pipeline.queue import parse, state
jobs = [j["job"] for j in parse(sys.argv[1])]
states = [state(j) for j in jobs]
failed = [j for j, s in zip(jobs, states) if s == "failed"]
print(f"  {sys.argv[1]}: {states.count('done')} done / {len(failed)} failed / {len(jobs)}")
for j in failed:
    print(f"    failed: {j}")
PY
done
echo "-- latest worker events"
for lg in "$EXP_ROOT"/logs/queue_*_gpu*.log; do
  [ -f "$lg" ] && echo "  $(basename "$lg"): $(grep -E '\] (START|DONE|FAILED)' "$lg" | tail -1)"
done
echo "-- disk"
df -h "$EXP_ROOT" | tail -1
