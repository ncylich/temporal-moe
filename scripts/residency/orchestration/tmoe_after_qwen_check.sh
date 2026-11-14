#!/usr/bin/env bash
until grep -q "apply_adapter qwen check DONE" /workspace/rerun-logs/apply_check_qwen.out 2>/dev/null; do sleep 60; done
if grep -q "EXACT" /workspace/rerun-logs/apply_check_qwen.out && ! grep -q "NOT EXACT" /workspace/rerun-logs/apply_check_qwen.out; then
  TMOE_PRIO=4 nohup bash /workspace/tmoe_online_smoke_qwen.sh > /workspace/rerun-logs/online_smoke_qwen.out 2>&1 & echo $! > /workspace/pids/online_smoke_qwen.pid
  echo "### qwen apply EXACT -> qwen parity smoke launched $(date -u +%H:%M)"
else echo "### qwen apply NOT exact -> smoke not launched; needs a fix"; fi
