#!/usr/bin/env bash
M=${1:?}; until grep -q "### think-on $M DONE" /workspace/rerun-logs/think_on_$M.out 2>/dev/null; do sleep 300; done
bash /workspace/tmoe_think_surface.sh $M > /workspace/rerun-logs/think_surface_$M.out 2>&1
