#!/usr/bin/env bash
until grep -q "qwen diag C DONE" /workspace/rerun-logs/qwen_diag_c.out 2>/dev/null; do sleep 20; done
TMOE_PRIO=3 nohup bash /workspace/tmoe_online_smoke_qwen.sh > /workspace/rerun-logs/online_smoke_qwen.out 2>&1 & echo $! > /workspace/pids/online_smoke_qwen.pid
echo "### qwen smoke (with logprob parity) launched after diag C $(date -u +%H:%M)"
