#!/usr/bin/env bash
until grep -q "hook diag DONE" /workspace/rerun-logs/hook_diag.out 2>/dev/null; do sleep 30; done
nohup bash /workspace/tmoe_qwen_finish.sh > /workspace/rerun-logs/qwen_finish.out 2>&1 & echo $! > /workspace/pids/qwen_finish.pid
echo "### qwen-finish chain relaunched after the hook diag $(date -u +%H:%M)"
