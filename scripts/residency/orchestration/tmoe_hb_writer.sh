#!/bin/bash
while true; do
  /workspace/temporal-moe/scripts/residency/heartbeat.sh >> /workspace/rerun-logs/heartbeat.log 2>&1
  echo "---" >> /workspace/rerun-logs/heartbeat.log
  sleep 3600
done
