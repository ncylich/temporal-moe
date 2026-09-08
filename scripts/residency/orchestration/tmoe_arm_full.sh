#!/bin/bash
# Re-measure one previously-trained arm on the FULL GSM8K split. Merges from the kept
# adapter if the merged dir was pruned, verifies the merge carries the trained surfaces,
# then scores 1319 problems. Every one of these arms was judged at n=200 against a base
# estimate now known to be wrong by 3 points, so none of them is actually falsified.
#   tmoe_arm_full.sh <suffix> <traj-tag>
set -euo pipefail
cd /workspace/temporal-moe
SUF=$1; TRAJ=$2
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-${SUF}-merged
A=/workspace/olmoe-adapt/data/gemma_ce_${SUF}_adapter.pt
[ -s "$A" ] || { echo "### no adapter $A"; exit 4; }
scripts/residency/disk_budget.sh || exit 3
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  --model $B --family gemma4 --no-unsloth --traj $TRAJ \
  --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_${SUF}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### ${SUF} n1319 DONE $(date -u +%H:%M)"
