#!/bin/bash
# FLAME 1e18 downstream lm-eval batch driver ("Part B").
# Loops the 20 1e18 checkpoints, builds per-run MODEL_ARGS EXACTLY matching each checkpoint's trained
# geometry (parsed from results/phase0/runs/<run>/run.meta), sets TEMPORAL correctly (1 for temporal
# runs -> native masked residency router, 0 for dense/moe), runs analysis/probes/run_lmeval.py 0-shot
# on 10 tasks, and appends parsed metrics to results/ablations/flame1e18_downstream.csv.
#
# Idempotent/resumable: a run whose rows already exist in the CSV is skipped.
#
# Usage:
#   flame1e18_downstream.sh                 # run all 20
#   flame1e18_downstream.sh <run_name>      # run a single named run (testing)
# Env overrides (for validation only):
#   TASKS=piqa,copa   override the 10-task list
#   LIMIT=8           pass --limit 8 to lm_eval (few examples)
#   OUT_TAG=foo       output subdir tag (default lmeval_1e18_0shot)
#   NO_CSV=1          do not append to the CSV (validation runs)
set -euo pipefail
ROOT="${TMOE_ROOT:-${TEMPORAL_MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
RUNS=$ROOT/results/phase0/runs
CSV=$ROOT/results/ablations/flame1e18_downstream.csv
RDZV_PORT=${RDZV_PORT:-29537}
OUT_TAG=${OUT_TAG:-lmeval_1e18_0shot}
TASKS=${TASKS:-arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande,boolq,copa,lambada_openai,sciq}

ALL_RUNS=(
  flame38m_dense_local flame38m_g1_moe flame38m_g1_moe_s2 flame38m_g1_moe_s3
  flame38m_g1_temporal flame38m_g1_temporal_s2 flame38m_g1_temporal_s3
  flame38m_g3_moe flame38m_g3_moe_s2 flame38m_g3_moe_s3
  flame38m_g3_temporal flame38m_g3_temporal_s2 flame38m_g3_temporal_s3
  flame192_g3_moe flame192_g3_temporal
  flame512_dense flame512_g1_moe flame512_g1_temporal flame512_g3_moe flame512_g3_temporal
)

# single-run override
if [ "$#" -ge 1 ]; then RUN_LIST=("$1"); else RUN_LIST=("${ALL_RUNS[@]}"); fi

meta_get() { echo "$1" | grep -oE "[[:space:]]$2=[^ ]+" | head -1 | sed 's/.*=//'; }

run_one() {
  local RUN=$1
  local OUT=$RUNS/$RUN
  local CKPT=$OUT/ckpt
  local META_F=$OUT/run.meta
  [ -f "$META_F" ] || { echo "[skip] $RUN: no run.meta"; return; }
  [ -d "$CKPT" ] || { echo "[skip] $RUN: no ckpt dir"; return; }

  # idempotent: skip if this model already has rows in the CSV
  if [ "${NO_CSV:-0}" != "1" ] && [ -f "$CSV" ] && grep -q "^$RUN," "$CSV"; then
    echo "[skip] $RUN: already in $CSV"; return
  fi

  local META; META=$(cat "$META_F")
  local MODE FFN NUM_EXPERTS TOPK MOE_FFN SHARED
  MODE=$(meta_get "$META" mode)
  FFN=$(meta_get "$META" ffn)
  NUM_EXPERTS=$(meta_get "$META" num_experts)
  TOPK=$(meta_get "$META" topk)
  MOE_FFN=$(meta_get "$META" moe_ffn)
  SHARED=$(meta_get "$META" shared)

  # H/heads from the size class (all L=9); heads = H/16
  local H
  case "$RUN" in
    flame38m_*) H=256 ;;
    flame192*)  H=192 ;;
    flame512*)  H=512 ;;
    *) echo "[skip] $RUN: unknown size class"; return ;;
  esac
  local HEADS=$((H/16))

  local TEMPORAL=0
  [ "$MODE" = temporal ] && TEMPORAL=1

  # MoE args only for moe/temporal
  local MOE_ARGS=()
  if [ "$MODE" != dense ]; then
    MOE_ARGS=(
      --moe-ffn-hidden-size "$MOE_FFN" --num-experts "$NUM_EXPERTS" --moe-router-topk "$TOPK"
      --moe-shared-expert-intermediate-size "$SHARED" --moe-layer-freq "[0]*1+[1]*8"
      --moe-router-dtype fp32 --moe-router-pre-softmax --moe-router-score-function softmax
      --moe-aux-loss-coeff 0.01 --moe-z-loss-coeff 0.001 --moe-grouped-gemm
    )
  fi
  local MODEL_ARGS=(
    --hidden-size "$H" --ffn-hidden-size "$FFN" --num-layers 9 --num-attention-heads "$HEADS"
    --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear
    "${MOE_ARGS[@]}"
    --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model EleutherAI/pythia-12b
    --transformer-impl transformer_engine --no-gradient-accumulation-fusion
  )
  local INFRA_ARGS=(
    --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --use-distributed-optimizer
  )
  [ "$MODE" != dense ] && INFRA_ARGS+=(--moe-token-dispatcher-type alltoall)

  local LIMIT_ARG=()
  [ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")

  echo "===================================================================="
  echo "[run] $RUN mode=$MODE TEMPORAL=$TEMPORAL H=$H heads=$HEADS ffn=$FFN experts=$NUM_EXPERTS topk=$TOPK moe_ffn=$MOE_FFN shared=$SHARED"
  echo "[run] tasks=$TASKS out=$OUT/$OUT_TAG"
  echo "===================================================================="

  cd "$ROOT/lm-evaluation-harness"
  # CUDA/cuDNN + Megatron runtime env (mirrors experiments/run.sh:96-107 and flame_scale_run.sh:48-53)
  export OMP_NUM_THREADS=16 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  local NV=/usr/local/lib/python3.11/dist-packages/nvidia
  export CUDNN_PATH=$NV/cudnn
  export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
  export PATH=$ROOT/.venv/bin:$PATH
  export HF_DATASETS_TRUST_REMOTE_CODE=true HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
  export TEMPORAL=$TEMPORAL TEMPORAL_EVICT=${TEMPORAL_EVICT:-min_logit}
  export PYTHONPATH=$ROOT/lm-evaluation-harness:$ROOT/Megatron-LM:${PYTHONPATH:-}

  "$(dirname "$PY")/torchrun" --nproc_per_node=1 --rdzv-endpoint=localhost:$RDZV_PORT \
    "$ROOT/analysis/probes/run_lmeval.py" \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" \
    --bf16 --seq-length 2048 --micro-batch-size 32 --batch_size "${LM_BATCH:-16}" \
    --max-tokens-to-oom 10000000 --seed 42 --load "$CKPT" --model megatron_lm \
    --num_fewshot 0 --tasks "$TASKS" "${LIMIT_ARG[@]}" \
    --output_path "$OUT/$OUT_TAG" \
    2>&1 | tee "$OUT/${OUT_TAG}.log"

  if [ "${NO_CSV:-0}" != "1" ]; then
    "$PY" "$ROOT/analysis/probes/flame1e18_downstream_csv.py" "$RUN" "$OUT/$OUT_TAG" "$CSV"
  fi
}

FAILED=()
for R in "${RUN_LIST[@]}"; do run_one "$R" || { echo "[FAIL] $R (continuing)"; FAILED+=("$R"); }; done
echo "[done] driver finished for: ${RUN_LIST[*]}"
[ ${#FAILED[@]} -gt 0 ] && echo "[done] FAILED runs: ${FAILED[*]}" || echo "[done] all runs succeeded"
