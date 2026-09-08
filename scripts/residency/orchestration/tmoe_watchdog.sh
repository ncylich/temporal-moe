#!/usr/bin/env bash
# Stall watchdog: emits a line (= wakes the session) when the newest log has been silent longer than
# STALL_MIN while the GPU is busy, when the GPU is idle with lease tickets waiting, or once when the GPU
# goes idle with nothing queued. Re-emits a persisting stall every STALL_MIN.
STALL_MIN=${STALL_MIN:-20}; last_idle_report=0; last_stall_report=0
while true; do
  sleep 300
  now=$(date +%s); used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); used=${used:-0}
  nq=$(ls /var/lock/tmoe_gpu0.q/ 2>/dev/null | wc -l)
  newest=$(ls -t /workspace/rerun-logs/*.out 2>/dev/null | head -1); age=$(( (now - $(stat -c %Y "$newest" 2>/dev/null || echo $now)) / 60 ))
  if [ "$used" -gt 10000 ]; then
    if [ "$age" -ge "$STALL_MIN" ] && [ $((now - last_stall_report)) -ge $((STALL_MIN*60)) ]; then
      echo "STALL $(date -u +%H:%M): GPU busy (${used} MiB) but $(basename $newest) silent for ${age} min; last: $(tail -1 $newest | cut -c1-100)"; last_stall_report=$now; fi
  else
    if [ "$nq" -gt 0 ] && [ "$age" -ge 10 ]; then
      echo "IDLE-WITH-QUEUE $(date -u +%H:%M): GPU idle, $nq ticket(s) waiting for ${age} min (RAM guard?) $(ls /var/lock/tmoe_gpu0.q/ | tr '\n' ' ')"
    elif [ "$nq" -eq 0 ] && [ $((now - last_idle_report)) -ge 3600 ]; then
      echo "GPU IDLE, NOTHING QUEUED $(date -u +%H:%M): last log $(basename $newest): $(grep -E '^### ' $newest | tail -1 | cut -c1-100)"; last_idle_report=$now; fi
  fi
done
