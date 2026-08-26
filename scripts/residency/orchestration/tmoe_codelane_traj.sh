#!/bin/bash
# Generate free-model trajectories for the new code lane, then splice them onto the
# existing D7 trajectory set. Only the new rows are generated; the 8,471 D7 rows are
# reused unchanged so the comparison is D7 vs D7+code, not two different pools.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$(ls /workspace/olmoe-adapt/data/codelane_*.jsonl | head -1)
echo "### codelane traj from $P $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/gen_traj_vllm.py \
  --model /dev/shm/gemma4-26b-it --tag gemma4_codelane --prompts "$P" \
  --max-new 1024 --max-prompt-tok 1024 --gpu-mem 0.90
echo "### codelane traj DONE $(date -u +%H:%M)"
