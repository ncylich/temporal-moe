#!/bin/bash
# Emit one line the moment ANY run in /workspace/rerun-logs finishes or fails.
#
# Written because per-file monitors only cover the jobs that existed when they were armed:
# lanes launched later (think-on, real-math, benchmarks) had no completion coverage, so a
# finished adapter sat unreported until the hourly heartbeat happened to mention it. This
# rescans the directory each pass, so new logs are picked up without re-arming anything.
#
# State lives in a file, so a marker is reported exactly once even across restarts.
STATE=/workspace/rerun-logs/.completion_seen
touch "$STATE"
PAT='### .*(ALL DONE|DONE [0-9]|TRAIN DONE|MERGE DONE|VERIFY DONE|REMEASURE DONE|PIPELINE COMPLETE|HUMANEVAL DONE)|^\[chain\] ABORT|Traceback \(most recent|CUDA out of memory|VERIFICATION FAILED|SMOKE FAIL'
while true; do
  for f in /workspace/rerun-logs/*.out /workspace/rerun-logs/*.log; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    grep -hoE "$PAT" "$f" 2>/dev/null | tail -3 | while read -r m; do
      key="$base :: $m"
      grep -qxF "$key" "$STATE" 2>/dev/null && continue
      echo "$key" >> "$STATE"
      echo "[$(date -u +%H:%M)] $base -> $m"
    done
  done
  sleep 60
done
