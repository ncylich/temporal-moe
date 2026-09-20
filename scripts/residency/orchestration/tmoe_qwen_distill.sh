#!/usr/bin/env bash
# FALLBACK #2: semi-on-policy self-distillation, one round (DAgger-shaped).
# The failure analysis says residency breaks arithmetic at generation time on the model's OWN
# prefixes; teacher-forced CE on the real pool sees digits as easy (loss 0.40 weighted). So:
#   1. sample the CONSTRAINED student (merged $SRC adapter, R8, eval's card recipe) on the D7 pool
#   2. label every sampled prefix with the FREE base model's top-50 logprobs (the teacher)
#   3. continue the $SRC adapter on KL(student constrained || teacher free) over those samples
# Same-arm R8 delta and false-equation rate vs the $SRC adapter are the readouts.
#   tmoe_qwen_distill.sh [rebuild|digit10]
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
SRC=${1:-rebuild}; NAME=distill_${SRC}
B=/root/models/qwen35-35b-a3b
SRCM=/root/models/qwen35-${SRC}-merged; SRCA=/workspace/olmoe-adapt/data/qwen_ce_${SRC}_adapter.pt
TRAJ=qwen35_selfgen_${SRC}R8; TP=/workspace/instruct-traj/$TRAJ.pt; KL=/workspace/instruct-traj/${TRAJ}_klref.pt
A=/workspace/olmoe-adapt/data/qwen_ce_${NAME}_adapter.pt; M=/root/models/qwen35-${NAME}-merged
L=scripts/residency/gpu_lease.sh
COMMON="--model $B --family qwen35 --no-unsloth --traj $TRAJ --max-seq 4096 --expert-lora-r 16 --opt adamw --micro-batch 16"
[ -d $SRCM ] && [ -s $SRCA ] || { echo "### missing $SRCM or $SRCA"; exit 2; }
scripts/residency/disk_budget.sh || exit 3
echo "### qwen-$NAME 1/5 sample student under R8 $(date -u +%H:%M)"
[ -s $TP ] || $L /workspace/venv_vllm312/bin/python -u analysis/residency/selfgen_traj.py \
  --path $SRCM --R 8 --prompts /workspace/olmoe-adapt/data/d7_prompts.jsonl --n 4500 --max-new 1024 \
  --think off --presence-penalty 1.5 --max-model-len 2560 --gpu-mem 0.85 --out $TP
echo "### qwen-$NAME 2/5 teacher logprobs (free base) $(date -u +%H:%M)"
[ -s $KL ] || $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON \
  --out /tmp/gce_precompute_unused.pt --precompute-kl $KL
echo "### qwen-$NAME 3/5 distill from $SRC adapter $(date -u +%H:%M)"
[ -f $A.done ] || { cp $SRCA $A
  $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --out $A --resume \
    --accum 16 --lr 3e-5 --tokens 5100000 --kl-only --kl-arm constrained --kl-anchor $KL --kl-weight 1.0
  touch $A.done; }
echo "### qwen-$NAME 4/5 merge $(date -u +%H:%M)"
[ -d $M ] || { $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py $COMMON --out $A --merge-out $M
  /workspace/venv_fla/bin/python analysis/residency/textify_qwen_merge.py $M; }
echo "### qwen-$NAME 5/5 eval $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model qwen35_instruct --path $M --arms free,R8,R32 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
  --record-as qwen35_ce_${NAME}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
echo "### qwen-$NAME ALL DONE $(date -u +%H:%M)"
