#!/usr/bin/env bash
# Poll running queue workers every 60 s for up to $1 seconds (default 1740) and print one line:
# "ANOMALY <reason>" as soon as something breaks, "ALLDONE" when every job is finished,
# otherwise "TICK" at the deadline. Remaining arguments: queue files (default configs/queues/[0-9]_*.txt).
source "$(dirname "$0")/env.sh"
R="$EXP_ROOT"; END=$(( $(date +%s) + ${1:-1740} )); shift || true
QUEUES=("${@:-configs/queues/[0-9]_*.txt}")
# shellcheck disable=SC2206
QUEUES=(${QUEUES[@]})
jobs_of() { grep -v '^#' "$1" | awk -F'|' 'NF{print ($8!="" ? $8 : ($5!="" ? $5 : $1))}'; }
nfail0=$(cat "$R"/logs/queue_*_gpu*.log 2>/dev/null | grep -c "FAILED")
while [ "$(date +%s)" -lt "$END" ]; do
  done_all=1
  for f in "${QUEUES[@]}"; do
    for n in $(jobs_of "$f"); do [ -f "$R/locks/$n/done" ] || done_all=0; done
  done
  [ "$done_all" = 1 ] && { echo "ALLDONE"; exit 0; }
  nfail=$(cat "$R"/logs/queue_*_gpu*.log 2>/dev/null | grep -c "FAILED")
  [ "$nfail" -gt "$nfail0" ] && { echo "ANOMALY new FAILED job"; exit 0; }
  nw=$(ps -eo args | grep -c "^[^ ]*python[0-9.]* -m trac.pipeline.queue")
  [ "$nw" -lt 1 ] && { echo "ANOMALY no queue worker alive"; exit 0; }
  orph=$(ps -eo pid,ppid,args | awk '$2==1 && /VLLM::EngineCore/' | wc -l)
  [ "$orph" -gt 0 ] && { echo "ANOMALY orphan vLLM engine"; exit 0; }
  avail=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
  [ "${avail:-99}" -lt 5 ] && { echo "ANOMALY only ${avail}G free under EXP_ROOT"; exit 0; }
  sleep 60
done
echo "TICK"
