#!/usr/bin/env bash
# when the first online run finishes, start the from-scratch reverse-KL run (3.4M tokens, every 16 x 256)
until grep -q "online_digit3_e16 ALL DONE" /workspace/rerun-logs/gemma_online.out; do sleep 30; done
TMOE_PRIO=3 nohup bash /workspace/tmoe_gemma_online.sh scratch 3400000 16 256 >> /workspace/rerun-logs/gemma_online_scratch.out 2>&1 & echo $! > /workspace/pids/gemma_online_scratch.pid
echo "### from-scratch online run launched $(date -u +%H:%M) (expect ~1h50m)"
