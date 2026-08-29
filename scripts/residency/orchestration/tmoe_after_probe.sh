#!/usr/bin/env bash
until grep -q "sampler cost DONE" /workspace/rerun-logs/sampler_cost2.out 2>/dev/null; do sleep 15; done
TMOE_PRIO=3 bash /workspace/tmoe_speed.sh qwen nopp 2>&1 | grep -E "^\[speed\]|^### "
