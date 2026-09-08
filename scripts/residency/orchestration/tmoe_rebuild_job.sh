#!/bin/bash
# Merge the faithful D7 published-recipe reconstruction, verify the merge actually
# carries the trained surfaces, then score the FULL GSM8K test split.
# Assumes CUDA_VISIBLE_DEVICES is set by the slot wrapper.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ninja lives in the vllm venv; torch builds inline CUDA extensions at engine
# init and dies with FileNotFoundError if it is not on PATH (2026-08-26).
export PATH=/workspace/venv_vllm312/bin:$PATH
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-rebuild-merged
A=/workspace/olmoe-adapt/data/gemma_ce_rebuild_adapter.pt
scripts/residency/disk_budget.sh || exit 3
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 \
  --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
export HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
exec /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_rebuild_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
