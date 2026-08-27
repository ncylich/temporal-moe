#!/usr/bin/env bash
# Same fix on the second model: gemma rebuild recipe (D7, r32, KL 0.05, 3.4M) + --digit-weight.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
W=${1:-10}; NAME=digit${W}
B=/dev/shm/gemma4-26b-it; A=/workspace/olmoe-adapt/data/gemma_ce_${NAME}_adapter.pt
M=/root/models/gemma4-${NAME}-merged; KL=/workspace/instruct-traj/gemma4_d7_seq4096_klref.pt
L=scripts/residency/gpu_lease.sh
COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A"
scripts/residency/disk_budget.sh || exit 3
echo "### gemma-$NAME train $(date -u +%H:%M)"
[ -s "$A" ] || $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON \
  --accum 16 --lr 3e-5 --tokens 3400000 --kl-anchor $KL --kl-weight 0.05 --digit-weight $W
echo "### gemma-$NAME merge $(date -u +%H:%M)"
[ -d $M ] || { $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py $COMMON --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
/workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### gemma-$NAME eval $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_${NAME}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### gemma-$NAME ALL DONE $(date -u +%H:%M)"
