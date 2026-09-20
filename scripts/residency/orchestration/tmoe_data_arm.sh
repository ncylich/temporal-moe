#!/usr/bin/env bash
# One data-lever experiment end to end on a single GPU: trajectories -> KL ref -> train ->
# merge -> verify -> GSM8K n=1319. Every stage waits for a free device. Scored as the
# same-arm R8 delta against gemma4_instruct_n1319; the rebuild sits at +3.1 and the
# published D12 at +6.0. Bar to beat the rebuild: +4.5 (eval floor + run spread).
set -euo pipefail
cd /workspace/temporal-moe
NAME=$1; PROMPTS=$2; shift 2
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/dev/shm/gemma4-26b-it; TAG=gemma4_$NAME
A=/workspace/olmoe-adapt/data/gemma_ce_${NAME}_adapter.pt
M=/root/models/gemma4-${NAME}-merged
KL=/workspace/instruct-traj/${TAG}_seq4096_klref.pt
L=scripts/residency/gpu_lease.sh
scripts/residency/disk_budget.sh || exit 3
echo "### $NAME trajectories $(date -u +%H:%M)"
[ -s /workspace/instruct-traj/${TAG}.pt ] || { $L /workspace/venv_vllm312/bin/python -u analysis/residency/gen_traj_vllm.py \
    --model $B --tag $TAG --prompts "$PROMPTS" --max-new 3072 --max-prompt-tok 1024 --gpu-mem 0.90; }
[ -s /workspace/instruct-traj/${TAG}_seq4096.pt ] || \
  /workspace/venv_fla/bin/python analysis/residency/cut_trajectories.py --tag $TAG --max-seq 4096
COMMON="--model $B --family gemma4 --no-unsloth --traj ${TAG}_seq4096 --max-seq 4096
        --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A"
echo "### $NAME KL precompute $(date -u +%H:%M)"
[ -s "$KL" ] || { $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --precompute-kl $KL; }
echo "### $NAME train $(date -u +%H:%M)"
[ -s "$A" ] || { $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON \
    --accum 16 --lr 3e-5 --tokens 3400000 --kl-anchor $KL --kl-weight 0.05 "$@"; }
echo "### $NAME merge $(date -u +%H:%M)"
[ -d $M ] || { $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py $COMMON --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### $NAME eval $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_${NAME}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### $NAME ALL DONE $(date -u +%H:%M)"
