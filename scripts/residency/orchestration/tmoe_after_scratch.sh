#!/usr/bin/env bash
# after the from-scratch online run: the gemma deadband surfaces (base rho=0.5, then the W=3 adapter rho=0.5)
until grep -q "online_scratch_e16 ALL DONE" /workspace/rerun-logs/gemma_online_scratch.out 2>/dev/null; do sleep 60; done
TMOE_PRIO=5 nohup bash /workspace/tmoe_deadband_surface.sh gemma 0.5 /dev/shm/gemma4-26b-it gemma4_base >> /workspace/rerun-logs/deadband_base.out 2>&1 & echo $! > /workspace/pids/deadband_base.pid
sleep 5
TMOE_PRIO=6 nohup bash /workspace/tmoe_deadband_surface.sh gemma 0.5 adapter:/workspace/olmoe-adapt/data/gemma_ce_digit3_adapter.pt gemma4_ce_digit3 >> /workspace/rerun-logs/deadband_digit3.out 2>&1 & echo $! > /workspace/pids/deadband_digit3.pid
echo "### deadband surfaces launched $(date -u +%H:%M) (expect ~1h each)"
