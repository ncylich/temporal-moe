#!/bin/bash
# Replicate the FLAME-MoE paper's smallest model (FLAME-MoE-38M-100M @ 1e18 FLOPs) with the temporal
# rolling-residency router, so its CE is directly comparable to the paper's reported number.
#
# Paper config (configs/model/flame-moe.sh + scripts/release/flame-moe-38m.sh + configs/train/flame-moe.sh):
#   hidden 256, 9 layers, ffn 1368, moe_ffn 176, 64 experts, top-6, shared-intermediate 2*moe_ffn (=352),
#   moe_layer_freq [0]*1+[1]*8, pythia-12b (50k) tokenizer, dclm data, seq 2048,
#   global-batch 1024, micro-batch 32, lr 3e-4 -> min 3e-5, WSD decay (warmup-frac 0.01, wsd-decay iters/10),
#   train-iters 2121 (= 4.45B tokens = 1e18 FLOPs).
# Single-GPU adaptations (numerically equivalent; see EVALUATION_METHODOLOGY): EP=1 + --moe-grouped-gemm
#   (vs paper EP=8), TE impl, --no-gradient-accumulation-fusion, head_dim 16 (256/16 heads).
# Temporal: pretrain_temporal.py installs the rolling-residency router (K=top-6 resident of 64);
#   TEMPORAL_EVICT default min_logit (the swept config). Metric = CE (paper reports CE; 50k vocab so
#   our 16k-BPE BPB is not comparable — we match the paper's tokenizer exactly here).
#
# Env: TRAIN_ITERS (default 2121; set small for a smoke), RUN_NAME, TEMPORAL_EVICT, CE_FUSION.
set -euo pipefail
cd "$(dirname "$0")/../.."

TRAIN_ITERS=${TRAIN_ITERS:-2121}
# DENSE=1: fully-dense IsoFLOP floor — all 9 layers dense SwiGLU with ffn=1422 so the dense non-embedding
# params equal the MoE's ACTIVE non-embedding params (12.20M), same 1e18 budget/tokens. No MoE, no temporal.
DENSE=${DENSE:-0}
if [ "$DENSE" = "1" ]; then FFN=1422; else FFN=1368; fi
RUN_NAME=${RUN_NAME:-flame38m_$([ "$DENSE" = 1 ] && echo dense || echo temporal_${TEMPORAL_EVICT:-min_logit})}
WSD_DECAY=$("$PY" -c "print(max(1,$TRAIN_ITERS//10))")
EVAL_INTERVAL=${EVAL_INTERVAL:-$TRAIN_ITERS}     # default: one eval at the end
WARMUP_FRAC=0.01

OUT=$ROOT/results/phase0/runs/$RUN_NAME
CKPT=$OUT/ckpt
mkdir -p "$OUT"
echo "[run] $RUN_NAME dense=$DENSE ffn=$FFN iters=$TRAIN_ITERS gb=1024 mb=${MICRO_BATCH:-8} lr=3e-4 WSD(decay=$WSD_DECAY) evict=${TEMPORAL_EVICT:-min_logit} tok=pythia-12b(50k)" | tee "$OUT/run.meta"

DATA_DIR=${DATA_DIR:-$ROOT/data/dclm_tokenized}
DATA_PATH=$(find "$DATA_DIR" -type f -name '*_text_document.bin' \
  -exec sh -c '[ -f "${1%.bin}.idx" ] && printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
[ -z "$DATA_PATH" ] && { echo "ERROR: no tokenized .bin in $DATA_DIR"; exit 1; }

export OMP_NUM_THREADS=16 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 50k-vocab logits are large; reduce fragmentation
export TEMPORAL_EVICT=${TEMPORAL_EVICT:-min_logit}

MOE_ARGS=()
if [ "$DENSE" != "1" ]; then
  MOE_ARGS=(
    --moe-ffn-hidden-size 176 --num-experts 64 --moe-router-topk 6
    --moe-shared-expert-intermediate-size 352 --moe-layer-freq "[0]*1+[1]*8"
    --moe-router-dtype fp32 --moe-router-pre-softmax --moe-router-score-function softmax
    --moe-aux-loss-coeff 0.01 --moe-z-loss-coeff 0.001 --moe-grouped-gemm
  )
fi
MODEL_ARGS=(
  --hidden-size 256 --ffn-hidden-size $FFN --num-layers 9 --num-attention-heads 16
  --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6
  --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear
  "${MOE_ARGS[@]}"
  --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model EleutherAI/pythia-12b
  --transformer-impl transformer_engine --no-gradient-accumulation-fusion
  ${CE_FUSION:+--cross-entropy-loss-fusion}
)
INFRA_ARGS=(
  --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --use-distributed-optimizer
  --distributed-timeout-minutes 30 --bf16
)
[ "$DENSE" != "1" ] && INFRA_ARGS+=(--moe-token-dispatcher-type alltoall)
TRAIN_ARGS=(
  --micro-batch-size ${MICRO_BATCH:-8} --global-batch-size 1024
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-warmup-fraction $WARMUP_FRAC
  --lr-wsd-decay-iters $WSD_DECAY --train-iters $TRAIN_ITERS
  --weight-decay 0.01 --clip-grad 1.0 --seed ${SEED:-1234}
)
DATA_ARGS=( --seq-length 2048 --data-path $DATA_PATH --split 90,5,5 )
LOG_ARGS=(
  --log-interval 5 --log-throughput
  --save "$CKPT" --save-interval $TRAIN_ITERS --load "$CKPT"
  --eval-interval $EVAL_INTERVAL --eval-iters ${EVAL_ITERS:-50}
)
[ "$DENSE" != "1" ] && LOG_ARGS+=(--moe-per-layer-logging)

# DENSE -> plain pretrain_gpt.py (no temporal router); else pretrain_temporal.py installs the router patch.
ENTRY=$ROOT/temporal/pretrain_temporal.py
[ "$DENSE" = "1" ] && ENTRY=pretrain_gpt.py
cd Megatron-LM
if [ "${PROBE:-0}" = "1" ]; then
  # Mechanistic router probe (see run.sh PROBE): load CKPT, log per-token routing on one fixed batch.
  export ROUTER_LOG_OUT=$OUT/router_log.pt
  $ROOT/.venv/bin/torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29520} \
    $ROOT/analysis/probes/router_probe.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" \
    --finetune --train-iters 6 --lr-wsd-decay-iters 1 --save-interval 100000 --eval-iters 1 \
    2>&1 | tee "$OUT/probe.log"
else
  $ROOT/.venv/bin/torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29520} $ENTRY \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" \
    2>&1 | tee "$OUT/train.log"
fi
