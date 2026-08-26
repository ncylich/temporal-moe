#!/bin/bash
# D7+code arm: does a code-weighted mix fix code residency damage?
# Code share goes 5.1% -> 26.7% of rows. Everything else matches the winning recipe
# (3.4M response-token budget, lr 3e-5, expert-LoRA r32 + attention r32 + router/norm,
# KL anchor 0.05, R=8 constraint active during training) so the contrast is the mix alone.
set -euo pipefail
cd /workspace/temporal-moe
SEED=${1:-0}
SFX=$([ "$SEED" = 0 ] && echo "" || echo "_s$SEED")
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/dev/shm/gemma4-26b-it
A=/workspace/olmoe-adapt/data/gemma_ce_d7code${SFX}_adapter.pt
M=/root/models/gemma4-d7code${SFX}-merged
KL=/workspace/instruct-traj/gemma4_d7code_klref.pt
COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7code
        --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A"
scripts/residency/disk_budget.sh || exit 3
[ -s "$KL" ] || { echo "### d7code$SFX KL precompute $(date -u +%H:%M)"
  /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --precompute-kl $KL; }
echo "### d7code$SFX train $(date -u +%H:%M)"
[ -s "$A" ] || /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py \
  $COMMON --accum 16 --lr 3e-5 --data-seed $SEED --tokens 3400000 --kl-anchor $KL --kl-weight 0.05
echo "### d7code$SFX merge $(date -u +%H:%M)"
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  $COMMON --merge-out $M; cp $B/processor_config.json $M/ 2>/dev/null || true; }
/workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### d7code$SFX MBPP $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/mbpp_gemma.py \
  --path $M --arms free,R8,R16 --tag gemma4_ce_d7code${SFX} --max-model-len 4096 --gpu-mem 0.90
echo "### d7code$SFX HumanEval $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/humaneval_gemma.py \
  --path $M --arms free,R8,R16 --tag gemma4_ce_d7code${SFX}
echo "### d7code$SFX GSM8K $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_d7code${SFX}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### d7code$SFX ALL DONE $(date -u +%H:%M)"
