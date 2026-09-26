#!/usr/bin/env bash
# relaunch the rebased sweep once the qwen smoke has finished (pass or fail); the smoke verdict is read separately
until grep -q "qwen diag DONE" /workspace/rerun-logs/qwen_diag.out 2>/dev/null; do sleep 30; done
cd /workspace/temporal-moe
nohup /workspace/venv_vllm312/bin/python -u analysis/residency/sweep_online.py --aux-loss revkl_full --prio 4 --best gemma4_ce_online_scratch_e16_lr1e-4_n1319 --cells lr2e-4,klT2,temp1.0,refresh8x128,budget6.8M > /workspace/rerun-logs/sweep_online.out 2>&1 & echo $! > /workspace/pids/sweep_online.pid
echo "### sweep runner relaunched after the qwen diag $(date -u +%H:%M)"
