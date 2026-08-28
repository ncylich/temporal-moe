#!/bin/bash
# Dose-response at FIXED budget: same 0.6M tokens, different mean response length.
# short640 (mean 363) already gave GSM8K R8 -4.5, the best of any arm. If response length
# is the mechanism, short1024 (mean 650) on the SAME budget should land worse -- closer to
# the -5.5/-6.0 that every 668-831-token arm produced. Budget, pool, prompts, rank, KL and
# batch are all held; only the length distribution moves.
set -euo pipefail
cd /workspace/temporal-moe
CUT=$1; SUF=$2
G=$(NEED_GB=100 TIMEOUT=7200 scripts/residency/wait_for_gpu.sh) || exit 1
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G
T=gemma4_d7_seq4096_short${CUT}
KL=/workspace/instruct-traj/${T}_klref.pt
A=/workspace/olmoe-adapt/data/gemma_ce_${SUF}_adapter.pt
COMMON="--model /dev/shm/gemma4-26b-it --family gemma4 --no-unsloth --traj $T
        --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A"
echo "### $SUF on GPU $G $(date -u +%H:%M)"
[ -s "$KL" ] || /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --precompute-kl $KL
/workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON \
  --accum 16 --lr 3e-5 --tokens 600000 --kl-anchor $KL --kl-weight 0.05
echo "### $SUF TRAIN DONE $(date -u +%H:%M)"
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-${SUF}-merged
scripts/residency/disk_budget.sh || exit 3
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py $COMMON --merge-out $M; cp $B/processor_config.json $M/ 2>/dev/null || true; }
/workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
export PATH=/workspace/venv_vllm312/bin:$PATH HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_${SUF} \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94
echo "### $SUF GSM8K DONE $(date -u +%H:%M)"
