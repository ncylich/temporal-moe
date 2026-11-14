#!/usr/bin/env bash
# ReMoE (Zhu et al.) at ITS setting: router-only finetune with the recency-reuse objective, residency OFF during
# training, our data (d7 trajectories, 3.4M CE tokens), lr swept over 3 points (router-only tolerates higher lr).
# Eval per lr: GSM8K n=1319 free (their operating point: 100% of experts resident) and R8 (bounded ablation).
# Pick by free GSM8K -> full surface on the FREE arm, and loads/token under a plain LRU cache of half the experts
# (cache_bias walker, lambda 0) for the swap axis.   tmoe_remoe_fair.sh <gemma|qwen>
set -uo pipefail; cd /workspace/temporal-moe
MODEL=$1
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
export TMOE_PRIO=${TMOE_PRIO:-4}
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; D=/workspace/olmoe-adapt/data
if [ "$MODEL" = gemma ]; then B=/dev/shm/gemma4-26b-it; COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16"
  G="--model gemma4_instruct --path $B"; ML=4096; PFX=gemma4_remoe; CB_C=64; ARMS=free,R8
else B=/root/models/qwen35-35b-a3b; COMMON="--model $B --family qwen35 --no-unsloth --traj qwen35_d7_seq4096 --max-seq 4096 --expert-lora-r 16 --opt adamw --micro-batch 16"
  G="--model qwen35_instruct --path $B --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5"; ML=5632; PFX=qwen35_remoe; CB_C=128; ARMS=free,R8; fi
for LR in 1e-4 3e-4 1e-3; do
  A=$D/${PFX}_lr${LR}_adapter.pt
  echo "### remoe $MODEL lr $LR train (router-only, residency off, 3.4M) $(date -u +%H:%M)"
  [ -f $A.done ] || { $L $PY -u analysis/residency/train_gemma_ce.py $COMMON --out $A --router-only --no-constraint --remoe-lambda 1.0 --remoe-gamma 0.9 --extra-lr-div 1 --accum 16 --lr $LR --tokens 3400000 && touch $A.done; }
  echo "### remoe $MODEL lr $LR GSM8K free,R8 n=1319 $(date -u +%H:%M)"
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --adapter $A --arms $ARMS --record-as ${PFX}_lr${LR}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len $ML --gpu-mem 0.90
done
PICK=$($PY - "$PFX" <<'PY'
import sys, csv
pfx = sys.argv[1]
lines = [l for l in open("results/ablations/instruct_genbench_vllm.csv") if not l.lstrip('"').startswith("#") and l.strip()]
rows = [r for r in csv.DictReader(lines) if r.get("task")]
best, bv = None, -1
for lr in ("1e-4", "3e-4", "1e-3"):
    v = [float(r["value"]) for r in rows if r["model"] == f"{pfx}_lr{lr}_n1319" and r["arm"] == "free" and r["task"] == "gsm8k_cot_zeroshot" and r["metric"] == "exact_match,flexible-extract"]
    if v: print(f"[remoe-pick] lr {lr}: free GSM8K {100*v[-1]:.1f}", file=sys.stderr)
    if v and v[-1] > bv: best, bv = lr, v[-1]
print(best or "")
PY
)
echo "### remoe $MODEL pick lr $PICK $(date -u +%H:%M)"; [ -n "$PICK" ] || exit 1
A=$D/${PFX}_lr${PICK}_adapter.pt
echo "### remoe $MODEL full surface, FREE arm (100% resident) $(date -u +%H:%M)"
TMOE_ARMS=free $L bash /workspace/tmoe_deadband_surface.sh $MODEL 0 adapter:$A ${PFX}_lr${PICK}
echo "### remoe $MODEL loads/token under a plain LRU cache of $CB_C experts (GSM8K) $(date -u +%H:%M)"
TEMPORAL_WALKER=cache_bias TEMPORAL_CB_C=$CB_C TEMPORAL_CB_LAMBDA=0 TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1 \
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --adapter $A --arms R8 --record-as ${PFX}_lr${PICK}_lruC${CB_C}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len $ML --gpu-mem 0.90
echo "### remoe $MODEL DONE $(date -u +%H:%M)"
