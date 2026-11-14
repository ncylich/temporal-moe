#!/usr/bin/env bash
# Replication seed of the digit-weight fix (data order differs; everything else identical).
# expert-LoRA r16, KL 0.1 free-arm anchor, lr 3e-5, 3.4M tokens) plus --digit-weight 10:
# digit tokens are 6.3% of the response, so at 10x they carry ~40% of the loss instead of
# 6%. If arithmetic slips are what residency breaks and what CE could not see, this is the
# minimal intervention that lets the adapter see them. Scored as same-arm R8 delta vs base.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
W=${1:-10}; SEED=${2:-1}; NAME=digit${W}s${SEED}
B=/root/models/qwen35-35b-a3b; A=/workspace/olmoe-adapt/data/qwen_ce_${NAME}_adapter.pt
M=/root/models/qwen35-${NAME}-merged; KL=/workspace/instruct-traj/qwen35_d7_seq4096_klref.pt
L=scripts/residency/gpu_lease.sh
COMMON="--model $B --family qwen35 --no-unsloth --traj qwen35_d7_seq4096 --max-seq 4096 --expert-lora-r 16 --opt adamw --micro-batch 16 --out $A"
scripts/residency/disk_budget.sh || exit 3
echo "### verify the standing digit10 merge $(date -u +%H:%M)"
[ -f /root/models/qwen35-digit10-merged/.verified ] || { $L /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged /root/models/qwen35-digit10-merged && touch /root/models/qwen35-digit10-merged/.verified; }
echo "### qwen-$NAME train $(date -u +%H:%M)"
[ -s "$A" ] || $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON \
  --accum 16 --lr 3e-5 --tokens 3400000 --kl-anchor $KL --kl-weight 0.1 --digit-weight $W --data-seed $SEED
echo "### qwen-$NAME merge $(date -u +%H:%M)"
[ -d $M ] || { $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py $COMMON --merge-out $M
  /workspace/venv_fla/bin/python analysis/residency/textify_qwen_merge.py $M
  $L /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M && touch $M/.verified; }
echo "### qwen-$NAME eval $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model qwen35_instruct --path $M --arms free,R8,R32 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
  --record-as qwen35_ce_${NAME}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
echo "### qwen-$NAME ALL DONE $(date -u +%H:%M)"
