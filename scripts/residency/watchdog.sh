#!/usr/bin/env bash
# Hourly heartbeat that does not depend on any agent process.
#
# Every status signal in this session so far has been a Monitor, which is a child of the agent. If
# the agent stops, the monitors stop, and the runs keep going with nothing recording that they did.
# This is a detached loop with no such coupling: it writes a line an hour into a tracked file and
# pushes it, so the branch itself is the record of what was running and when.
#
# It never touches the GPU and never starts or stops work. Reporting only.
#
#   nohup bash scripts/residency/watchdog.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=results/ablations/overnight_heartbeat.log
DEADLINE=$(date -u -d "2026-08-03 18:00:00" +%s)     # 11:00 PST, the stated end of autonomous work

while true; do
  now=$(date -u +%s)
  job=$(pgrep -af "analysis/residency/(train_ple|downstream)\.py" 2>/dev/null \
        | grep -v "bash -c" | sed 's/.*--tag \([^ ]*\).*/\1/;s/.*--csurf \([^ ]*\).*/ds:\1/' | head -1)
  cell=$(ls -t /workspace/olmoe-adapt/ce_free_*.log 2>/dev/null | head -1)
  last=$(grep -aE '^\[eval\]' "$cell" 2>/dev/null | tail -1 | sed 's/^\[eval\] //' | cut -c1-90)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  dirty=$(git status --porcelain results/ 2>/dev/null | grep -v overnight_heartbeat | wc -l)
  left=$(( (DEADLINE - now) / 60 ))
  printf '%s | job=%-34s | gpu=%-14s | uncommitted=%s | %dmin to deadline | %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${job:-NONE}" "$gpu" "$dirty" "$left" "${last:-no eval yet}" \
    >> "$LOG"

  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A "$LOG" >/dev/null 2>&1
  git commit -q -m "heartbeat $(date -u +%H:%MZ): ${job:-idle}" >/dev/null 2>&1
  git push -q origin ple-adaptation >/dev/null 2>&1

  # Stop an hour past the deadline: by then either the queue finished or it is stuck, and either
  # way an unattended loop appending forever is not helping anyone.
  [ "$now" -gt $((DEADLINE + 3600)) ] && {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) | watchdog exiting, past deadline" >> "$LOG"
    git add -A "$LOG" >/dev/null 2>&1
    git commit -q -m "heartbeat: watchdog exit" >/dev/null 2>&1
    git push -q origin ple-adaptation >/dev/null 2>&1
    exit 0
  }
  sleep 3600
done
