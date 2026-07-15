#!/bin/bash
# Generalized FLAME-MoE-38M-100M @ 1e18 runner: dense floor / full MoE / temporal, at G1 (6/64) or
# G3 (18/192) granularity. Identical to flame38m_temporal.sh in every non-MoE-granularity respect
# (256h/9L, gb 1024, mb 8, WSD lr 3e-4->3e-5, 2121 iters = 4.45B tok = 1e18, pythia-50k, CE, seed 1234,
# no CE-fusion). Fine-graining is compute-preserving: experts x GRAIN, top-k x GRAIN, moe_ffn/GRAIN
# (even), shared expert UNCHANGED (352 = 2*176). MoE FLOPs/active params ~fixed vs G1.
#
# Modes (pick one):  DENSE=1  -> dense floor (pretrain_gpt, no experts, ffn=1422)
#                    MOE_FULL=1 -> full MoE (pretrain_gpt WITH experts)
#                    (default) -> temporal (pretrain_temporal, rolling-residency router)
# Knobs: GRAIN (default 1), MICRO_BATCH (default 8), TRAIN_ITERS (default 2121), TEMPORAL_EVICT
#        (default min_logit), RUN_NAME, DATA_DIR (default data/dclm_tokenized, the 50k corpus).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

TRAIN_ITERS=${TRAIN_ITERS:-2121}
GRAIN=${GRAIN:-1}
DENSE=${DENSE:-0}
MOE_FULL=${MOE_FULL:-0}
MICRO_BATCH=${MICRO_BATCH:-8}
TEMPORAL_EVICT=${TEMPORAL_EVICT:-min_logit}

# fine-grained MoE knobs (shared expert intermediate stays 352 = 2*176, unchanged)
NUM_EXPERTS=$((64 * GRAIN))
TOPK=$((6 * GRAIN))
MOE_FFN=$(.venv/bin/python -c "print(2*round((176/$GRAIN)/2))")
SHARED_INT=352

if [ "$DENSE" = "1" ]; then FFN=1422; MODE=dense
elif [ "$MOE_FULL" = "1" ]; then FFN=1368; MODE=moe
else FFN=1368; MODE=temporal; fi
RUN_NAME=${RUN_NAME:-flame38m_g${GRAIN}_${MODE}$([ "$MODE" = temporal ] && echo _${TEMPORAL_EVICT})}
WSD_DECAY=$(.venv/bin/python -c "print(max(1,$TRAIN_ITERS//10))")
EVAL_INTERVAL=${EVAL_INTERVAL:-$TRAIN_ITERS}
WARMUP_FRAC=0.01

OUT=$ROOT/results/phase0/runs/$RUN_NAME
CKPT=$OUT/ckpt
mkdir -p "$OUT"
echo "[run] $RUN_NAME mode=$MODE grain=$GRAIN ffn=$FFN num_experts=$NUM_EXPERTS topk=$TOPK moe_ffn=$MOE_FFN shared=$SHARED_INT iters=$TRAIN_ITERS gb=1024 mb=$MICRO_BATCH lr=3e-4 WSD(decay=$WSD_DECAY) evict=$TEMPORAL_EVICT tok=pythia-12b(50k)" | tee "$OUT/run.meta"

DATA_DIR=${DATA_DIR:-$ROOT/data/dclm_tokenized}
DATA_PATH=$(find "$DATA_DIR" -type f -name '*_text_document.bin' \
  -exec sh -c '[ -f "${1%.bin}.idx" ] && printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
[ -z "$DATA_PATH" ] && { echo "ERROR: no tokenized .bin in $DATA_DIR"; exit 1; }

export OMP_NUM_THREADS=16 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 50k-vocab logits are large; reduce fragmentation
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export CUDNN_PATH=$NV/cudnn
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PATH=$ROOT/.venv/bin:$PATH
export TEMPORAL_EVICT

MOE_ARGS=()
if [ "$MODE" != "dense" ]; then
  MOE_ARGS=(
    --moe-ffn-hidden-size $MOE_FFN --num-experts $NUM_EXPERTS --moe-router-topk $TOPK
    --moe-shared-expert-intermediate-size $SHARED_INT --moe-layer-freq "[0]*1+[1]*8"
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
[ "$MODE" != "dense" ] && INFRA_ARGS+=(--moe-token-dispatcher-type alltoall)
TRAIN_ARGS=(
  --micro-batch-size $MICRO_BATCH --global-batch-size 1024
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
[ "$MODE" != "dense" ] && LOG_ARGS+=(--moe-per-layer-logging)

# temporal -> pretrain_temporal.py (installs router patch); dense/moe -> plain pretrain_gpt.py
ENTRY=pretrain_gpt.py
[ "$MODE" = "temporal" ] && ENTRY=$ROOT/temporal/pretrain_temporal.py
cd Megatron-LM
$ROOT/.venv/bin/torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29520} $ENTRY \
  "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"
