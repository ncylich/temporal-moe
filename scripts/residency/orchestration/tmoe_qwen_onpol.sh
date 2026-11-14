#!/usr/bin/env bash
# Self-distillation ROUND 2 (semi-on-policy, qwen). Round 1 (KL-only continuation on the pool's
# first 4500 prompts) failed for two reasons that are fixed here: the samples were the prose lane
# only (0.7% digit chars, 1.5 '=' per sample) and the KL-only phase dropped the free-arm anchor.
#   1. sample the W=3 student under R8 on a math/few-shot-weighted prompt set (mathlane_v2=2341,d5_fewshot=1183,domain8k=1000)
#   2. label with the free base's top-50 logprobs
#   3. train FROM SCRATCH with the W=3 recipe unchanged (CE + free-arm anchor) PLUS an on-policy
#      constrained-arm KL term on the samples (--aux-traj), same 3.4M budget as digit3.
# Readout: same-arm vs base and vs digit3 at n=1319.   tmoe_qwen_onpol.sh [digit3]
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
SRC=${1:-digit3}; NAME=onpol3
B=/root/models/qwen35-35b-a3b; SRCM=/root/models/qwen35-${SRC}-merged
TRAJ=qwen35_selfgen_${SRC}R8q; TP=/workspace/instruct-traj/$TRAJ.pt; AKL=/workspace/instruct-traj/${TRAJ}_klref.pt
KL=/workspace/instruct-traj/qwen35_d7_seq4096_klref.pt
A=/workspace/olmoe-adapt/data/qwen_ce_${NAME}_adapter.pt; M=/root/models/qwen35-${NAME}-merged
L=scripts/residency/gpu_lease.sh
COMMON="--model $B --family qwen35 --no-unsloth --max-seq 4096 --expert-lora-r 16 --opt adamw --micro-batch 16"
[ -d $SRCM ] || { echo "### missing $SRCM"; exit 2; }
scripts/residency/disk_budget.sh || exit 3
echo "### qwen-$NAME 1/5 sample $SRC student under R8, lane quota mathlane_v2=2341,d5_fewshot=1183,domain8k=1000 $(date -u +%H:%M)"
[ -s $TP ] || $L /workspace/venv_vllm312/bin/python -u analysis/residency/selfgen_traj.py \
  --path $SRCM --R 8 --prompts /workspace/olmoe-adapt/data/d7_prompts.jsonl --quota "mathlane_v2=2341,d5_fewshot=1183,domain8k=1000" --max-new 1024 \
  --think off --presence-penalty 1.5 --max-model-len 2560 --gpu-mem 0.85 --out $TP
echo "### qwen-$NAME 2/5 teacher logprobs on the samples $(date -u +%H:%M)"
[ -s $AKL ] || $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --traj $TRAJ \
  --out /tmp/gce_precompute_unused.pt --precompute-kl $AKL
echo "### qwen-$NAME 3/5 train from scratch: W=3 recipe + on-policy KL $(date -u +%H:%M)"
[ -s $A ] || $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --traj qwen35_d7_seq4096 --out $A \
  --accum 16 --lr 3e-5 --tokens 3400000 --kl-anchor $KL --kl-weight 0.1 --digit-weight 3 \
  --aux-traj $TRAJ --aux-kl-anchor $AKL --aux-kl-weight 1.0
echo "### qwen-$NAME 4/5 merge $(date -u +%H:%M)"
[ -d $M ] || { $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py $COMMON --traj qwen35_d7_seq4096 --out $A --merge-out $M
  /workspace/venv_fla/bin/python analysis/residency/textify_qwen_merge.py $M
  $L /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M; }
echo "### qwen-$NAME 5/5 eval $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model qwen35_instruct --path $M --arms free,R8,R32 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
  --record-as qwen35_ce_${NAME}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
echo "### qwen-$NAME ALL DONE $(date -u +%H:%M)"
