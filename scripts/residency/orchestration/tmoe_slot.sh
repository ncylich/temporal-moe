#!/bin/bash
# Serialize GPU claims with a per-device lock. Polling free memory races: two runs both
# observe a device idle, both start, and the second dies in vLLM's memory-utilization
# check while the first is still allocating (2026-08-26: lost both full-split runs this
# way). flock holds the device for the whole job, not just the poll.
# Usage: tmoe_slot.sh <logfile> <cmd...>   -- all four GPUs available (2026-08-26).
set -euo pipefail
LOG=$1; shift
while :; do
  for g in 0 1 2 3; do
    exec {fd}>/var/lock/tmoe_gpu${g}.lock
    if flock -n $fd; then
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ "$used" -lt 2000 ]; then
        echo "### slot: GPU $g acquired $(date -u +%H:%M)" >> "$LOG"
        CUDA_VISIBLE_DEVICES=$g "$@" >> "$LOG" 2>&1
        rc=$?
        echo "### slot: GPU $g released rc=$rc $(date -u +%H:%M)" >> "$LOG"
        exit $rc
      fi
      flock -u $fd
    fi
    exec {fd}>&-
  done
  sleep 45
done
