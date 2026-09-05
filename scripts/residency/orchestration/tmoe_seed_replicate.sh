#!/bin/bash
# Training-variance replicate of the rebuild recipe. Identical in every respect except
# --data-seed, which permutes batch order. The paired McNemar bar on GSM8K covers question
# sampling only; it says nothing about run-to-run spread, so a single run beating base at
# z=2.97 is an upper bound on confidence. Two replicates give the between-run spread that
# the claim actually needs.
#   tmoe_seed_replicate.sh <seed>
set -euo pipefail
cd /workspace/temporal-moe
SEED=$1
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/dev/shm/gemma4-26b-it
A=/workspace/olmoe-adapt/data/gemma_ce_seed${SEED}_adapter.pt
M=/root/models/gemma4-seed${SEED}-merged
KL=/workspace/instruct-traj/gemma4_d7_seq4096_klref.pt
COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096
        --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A"
scripts/residency/disk_budget.sh || exit 3
echo "### seed$SEED train $(date -u +%H:%M)"
[ -s "$A" ] || /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py \
  $COMMON --data-seed $SEED --accum 16 --lr 3e-5 --tokens 3400000 \
  --kl-anchor $KL --kl-weight 0.05
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  $COMMON --merge-out $M; cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_seed${SEED}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### seed$SEED n1319 DONE $(date -u +%H:%M)"
