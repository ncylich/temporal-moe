#!/bin/bash
# Respawns the heartbeat writer if anything kills it. Checks every 5 minutes.
while true; do
  if ! ps -eo cmd | grep -q "[t]moe_hb_writer"; then
    setsid /workspace/tmoe_hb_writer.sh >/dev/null 2>&1 &
    echo "$(date -u '+%H:%M') supervisor: respawned heartbeat writer" \
      >> /workspace/rerun-logs/heartbeat.log
  fi
  sleep 300
done
