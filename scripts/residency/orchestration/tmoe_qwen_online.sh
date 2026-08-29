#!/usr/bin/env bash
# qwen3.5 replication of the on-policy reverse-KL recipe (no digit weight). Mirrors tmoe_gemma_online.sh.
#   tmoe_qwen_online.sh [start=scratch|digit3|rebuild] [tokens=3400000] [every=16] [n=256]
# Expectation: qwen trains ~433 tok/s (3x slower than gemma) -> 3.4M tokens ~2h10m + refreshes (~60 s each
# at 256 rows; ~16 refreshes) ~16 min + boot 5 min; eval (adapter-direct, free/R8/R32) ~25 min.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)} HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
START=${1:-scratch}; T=${2:-3400000}; EVERY=${3:-16}; N=${4:-256}; NAME=online_${START}_e${EVERY}${TMOE_NAME_SUFFIX:-}
B=/root/models/qwen35-35b-a3b; L=scripts/residency/gpu_lease.sh; D=/workspace/olmoe-adapt/data; PY=/workspace/venv_vllm312/bin/python
KL=/workspace/instruct-traj/qwen35_d7_seq4096_klref.pt
# same switches as tmoe_gemma_online.sh; defaults = the from-scratch formulation (anchor 0, lr 1e-4 = gemma's sweep best)
case "${TMOE_ANCHOR_W:-0}" in 0|0.0) ANCHOR_ARGS="";; *) ANCHOR_ARGS="--kl-anchor $KL --kl-weight ${TMOE_ANCHOR_W}";; esac
if [ -n "${TMOE_CE:-}" ]; then CE_ARGS="--digit-weight ${TMOE_W:-3}"; else CE_ARGS="--kl-only"; fi
COMMON="--model $B --family qwen35 --no-unsloth --traj qwen35_d7_seq4096 --max-seq 4096 --expert-lora-r 16 --opt adamw --micro-batch 16"
A=$D/qwen_ce_${NAME}_adapter.pt
if [ "$START" = scratch ]; then SEEN=0; INIT=""; else
  SEEN=$($PY -c "import torch,sys; print(int(torch.load(sys.argv[1], weights_only=False, map_location='cpu')['seen']))" $D/qwen_ce_${START}_adapter.pt); INIT="--resume"; fi
scripts/residency/disk_budget.sh || exit 3
echo "### qwen-$NAME 1/2 online reverse-KL from $START (seen=$SEEN -> $((SEEN+T))), refresh every $EVERY steps x $N rows $(date -u +%H:%M)"
[ -f $A.done ] || { [ -n "$INIT" ] && cp $D/qwen_ce_${START}_adapter.pt $A
  $L $PY -u analysis/residency/train_gemma_ce.py $COMMON --out $A $INIT --accum 16 --lr ${TMOE_LR:-1e-4} --tokens $((SEEN+T)) \
    $CE_ARGS $ANCHOR_ARGS --aux-loss ${TMOE_AUX_LOSS:-revkl_full} --aux-kl-weight ${TMOE_AUX_W:-1.0} --aux-kl-temp ${TMOE_KL_TEMP:-1.0} \
    --online-every $EVERY --online-n $N --online-max-new 1024 --online-quota "${TMOE_QUOTA:-mathlane_v2=2341,d5_fewshot=1183,domain8k=1000}" --online-temp ${TMOE_ONLINE_TEMP:-0.7} --budget-on ${TMOE_BUDGET_ON:-sampled} \
    --online-gpu-mem ${TMOE_ONLINE_MEM:-0.55} --online-offload ${TMOE_ONLINE_OFFLOAD:-20} --online-presence-penalty ${TMOE_ONLINE_PP:-1.5}
  touch $A.done; }
echo "### qwen-$NAME 2/2 GSM8K n=1319 via apply_adapter $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path $B --adapter $A --arms free,R8,R32 --think off \
  --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 --record-as qwen35_ce_${NAME}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
echo "### qwen-$NAME ALL DONE $(date -u +%H:%M)"
