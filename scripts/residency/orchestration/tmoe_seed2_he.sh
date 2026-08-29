#!/usr/bin/env bash
# Third run on HumanEval@8192 + MBPP@8192. rebuild and seed3 disagree by 5.4 points on
# HumanEval (n=164); a third run says which is the outlier. seed2 is the other D7 seed
# with its adapter on disk; it needs a re-merge (its merge was pruned).
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-seed2-merged
A=/workspace/olmoe-adapt/data/gemma_ce_seed2_adapter.pt; L=scripts/residency/gpu_lease.sh
scripts/residency/disk_budget.sh || exit 3
[ -d $M ] || { echo "### seed2 merge $(date -u +%H:%M)"
  $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
    --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 \
    --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### seed2 HumanEval@8192 $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/humaneval_gemma.py \
  --path $M --arms free,R8,R16 --tag gemma4_ce_seed2_he8192 --max-tokens 8192 --max-model-len 9216
echo "### seed2 MBPP@8192 $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/mbpp_gemma.py \
  --path $M --arms free,R8,R16 --tag gemma4_ce_seed2_m8192 --max-tokens 8192 --max-model-len 9216 --gpu-mem 0.90
echo "### seed2 CODE DONE $(date -u +%H:%M)"
