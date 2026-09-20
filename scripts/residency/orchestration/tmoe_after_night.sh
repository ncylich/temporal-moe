#!/usr/bin/env bash
until grep -q "night ALL DONE" /workspace/rerun-logs/night.out 2>/dev/null; do sleep 300; done
echo "### after-night: ReMoE at its setting, gemma then qwen $(date -u +%H:%M)"
bash /workspace/tmoe_remoe_fair.sh gemma > /workspace/rerun-logs/remoe_fair_gemma.out 2>&1
bash /workspace/tmoe_remoe_fair.sh qwen > /workspace/rerun-logs/remoe_fair_qwen.out 2>&1
echo "### after-night ALL DONE $(date -u +%H:%M)"
