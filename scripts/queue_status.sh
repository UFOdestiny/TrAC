#!/usr/bin/env bash
# Read-only report on queue workers: active stages, GPUs, per-matrix progress, failed jobs, disk.
#   bash scripts/queue_status.sh [matrices, default: main new_backbone]
source "$(dirname "$0")/env.sh"
if [ "$#" -gt 0 ]; then
  MATRICES=("$@")
else
  MATRICES=(main new_backbone)
fi
echo "== $(date '+%F %T')"
echo "-- active stages"
ps -eo pid,etime,args | grep -E "python -m (rewind|baselines|analysis)\.pipeline\." | grep -v grep | cut -c1-150
echo "-- GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null
echo "-- progress (done / failed / total jobs)"
for m in "${MATRICES[@]}"; do
  "$PY" - "$m" <<'PY'
import sys
from rewind.pipeline.queue import jobs, state
names = [j["job"] for j in jobs(sys.argv[1])]
states = [state(j) for j in names]
failed = [j for j, s in zip(names, states) if s == "failed"]
print(f"  {sys.argv[1]}: {states.count('done')} done / {len(failed)} failed / {len(names)}")
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
