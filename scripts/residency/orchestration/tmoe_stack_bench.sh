#!/bin/bash
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
COMMON="--model /dev/shm/gemma4-26b-it --traj gemma4_d7_seq4096 --max-seq 4096
        --expert-lora-r 32 --family gemma4 --opt adamw --micro-batch 16
        --out /workspace/olmoe-adapt/data/bench_probe.pt --smoke"
echo "########## HF+peft (what we run today) ##########"
/workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --no-unsloth 2>&1 | grep -E "STEADY|SMOKE PASS|parity"
echo "########## unsloth + deterministic algorithms ##########"
TORCH_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON 2>&1 | grep -E "STEADY|SMOKE PASS|parity|AssertionError"
