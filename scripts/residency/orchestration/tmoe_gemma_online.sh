#!/usr/bin/env bash
# TRUE on-policy distillation, in-process: every N steps the trainer syncs the CURRENT adapter into a
# sleeping vLLM engine on the same GPU, samples fresh trajectories under R8, labels them with the
# frozen base in-process, and trains the reverse-KL sampled-token objective (+ free-arm anchor,
# no CE) on them. Starts from the W=3 adapter; +T tokens; merge; verify; GSM8K n=1319.
#   tmoe_gemma_online.sh [start=digit3] [tokens=850000] [every=16] [n=256]
# Expectation: ~800 tok/s training plus ~2 min per refresh (16 steps ~ 3.7 min) => ~1.5x the offline
# training rate, no separate sampling stage: 0.85M tokens in ~30 min, +12 min merge/verify/eval.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)} HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
START=${1:-digit3}; T=${2:-850000}; EVERY=${3:-16}; N=${4:-256}; NAME=online_${START}_e${EVERY}
# START=scratch: no adapter, the base under R8 is the initial student; distillation-only from the ground up
B=/dev/shm/gemma4-26b-it; L=scripts/residency/gpu_lease.sh; D=/workspace/olmoe-adapt/data; PY=/workspace/venv_vllm312/bin/python
KL=/workspace/instruct-traj/gemma4_d7_seq4096_klref.pt
COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16"
A=$D/gemma_ce_${NAME}_adapter.pt; M=/root/models/gemma4-${NAME}-merged
if [ "$START" = scratch ]; then SEEN=0; INIT=""; else
  SEEN=$($PY -c "import torch,sys; print(int(torch.load(sys.argv[1], weights_only=False, map_location='cpu')['seen']))" $D/gemma_ce_${START}_adapter.pt); INIT="--resume"; fi
scripts/residency/disk_budget.sh || exit 3
echo "### $NAME 1/3 online reverse-KL from $START (seen=$SEEN -> $((SEEN+T))), refresh every $EVERY steps x $N rows $(date -u +%H:%M)"
[ -f $A.done ] || { [ -n "$INIT" ] && cp $D/gemma_ce_${START}_adapter.pt $A; rm -f $A.tmp
  $L $PY -u analysis/residency/train_gemma_ce.py $COMMON --out $A $INIT --accum 16 --lr 3e-5 --tokens $((SEEN+T)) \
    --kl-only --kl-anchor $KL --kl-weight 0.05 --aux-loss revkl --aux-kl-weight 1.0 \
    --online-every $EVERY --online-n $N
  touch $A.done; }
if [ "${TMOE_MERGE:-0}" = 1 ]; then
  echo "### $NAME 2/3 merge + verify $(date -u +%H:%M)"
  [ -d $M ] || { $L $PY analysis/residency/train_gemma_ce.py $COMMON --out $A --merge-out $M; cp $B/processor_config.json $M/ 2>/dev/null || true; }
  $L $PY analysis/residency/verify_merge.py --base $B --merged $M
  EVAL_MODEL="--path $M"
else
  echo "### $NAME 2/3 no merge: the eval engine applies the adapter directly (apply_adapter.py, bit-exact, ~8 s) $(date -u +%H:%M)"
  EVAL_MODEL="--path $B --adapter $A"
fi
echo "### $NAME 3/3 GSM8K n=1319 $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct $EVAL_MODEL --arms free,R8,R16 --record-as gemma4_ce_${NAME}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### $NAME ALL DONE $(date -u +%H:%M)"
