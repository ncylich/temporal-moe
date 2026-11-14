#!/usr/bin/env bash
until grep -q "wb finals ALL DONE" /workspace/rerun-logs/wb_finals.out 2>/dev/null; do sleep 300; done
echo "### after-wb: thinking-on adaptation, gemma then qwen $(date -u +%H:%M)"
bash /workspace/tmoe_think_on.sh gemma > /workspace/rerun-logs/think_on_gemma.out 2>&1
bash /workspace/tmoe_think_on.sh qwen > /workspace/rerun-logs/think_on_qwen.out 2>&1
echo "### after-wb ALL DONE $(date -u +%H:%M)"
