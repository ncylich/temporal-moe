#!/usr/bin/env bash
until grep -q "\[gce\] DONE seen=" /workspace/rerun-logs/gemma_onpol.out; do sleep 20; done
sleep 10; touch /workspace/olmoe-adapt/data/gemma_ce_onpol3_adapter.pt.done; echo "### mix training done $(date -u +%H:%M); .done touched" >> /workspace/rerun-logs/gemma_onpol.out
nohup bash /workspace/tmoe_gemma_onpol.sh digit3 mix >> /workspace/rerun-logs/gemma_onpol.out 2>&1 & echo $! > /workspace/pids/gemma_onpol.pid
