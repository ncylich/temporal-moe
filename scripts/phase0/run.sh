#!/bin/bash
# Single-GPU FLAME-MoE Phase-0 launcher (stock FLAME-MoE, B=1, EP=1, torchrun nproc=1).
# Adapted from scripts/training/flame-moe.sh: no SLURM, no GCS, local transformer impl.
# All knobs via env vars (see defaults). Computes train_iters so C = 6*N*D hits TARGET_FLOPS.
set -euo pipefail
cd "$(dirname "$0")/../.."        # repo root
ROOT=$(pwd)

# ---- run config (env overridable) ----
SHAPE=${SHAPE:?set SHAPE s1..s6}
TARGET_FLOPS=${TARGET_FLOPS:?set TARGET_FLOPS e.g. 1e17}
PEAK_LR=${PEAK_LR:-3e-3}
WARMUP_FRAC=${WARMUP_FRAC:-0.05}
GLOBAL_BATCH=${GLOBAL_BATCH:-256}
MICRO_BATCH=${MICRO_BATCH:-32}
SEED=${SEED:-1234}
AUX_COEFF=${AUX_COEFF:-0.01}
RUN_NAME=${RUN_NAME:-${SHAPE}_${TARGET_FLOPS}_lr${PEAK_LR}_wu${WARMUP_FRAC}_gb${GLOBAL_BATCH}_s${SEED}}
EXTRA_ARGS=${EXTRA_ARGS:-}

# ---- shape geometry ----
case $SHAPE in
  s1) H=192; L=5;  FFN=1026; MOE_FFN=132;;
  s2) H=256; L=6;  FFN=1368; MOE_FFN=176;;
  s3) H=320; L=7;  FFN=1710; MOE_FFN=220;;
  s4) H=384; L=8;  FFN=2052; MOE_FFN=264;;
  s5) H=448; L=9;  FFN=2394; MOE_FFN=308;;
  s6) H=512; L=10; FFN=2736; MOE_FFN=352;;
  *) echo "bad SHAPE $SHAPE"; exit 1;;
esac
MOE_LAYER_FREQ="[0]+[1]*$((L-1))"
SHARED_INT=$((2 * MOE_FFN))

# ---- compute iters so C = 6*N*D ----
read N TRAIN_ITERS < <(.venv/bin/python scripts/phase0/shapes.py iters "$SHAPE" "$TARGET_FLOPS" "$GLOBAL_BATCH")
WARMUP_ITERS=$(.venv/bin/python -c "print(max(1,round($WARMUP_FRAC*$TRAIN_ITERS)))")
MIN_LR=$(.venv/bin/python -c "print($PEAK_LR*0.1)")
EVAL_INTERVAL=$(.venv/bin/python -c "print(max(1,round($TRAIN_ITERS/10)))")   # 1e16 point = iters/10
SAVE_INTERVAL=$EVAL_INTERVAL

OUT=$ROOT/results/phase0/runs/$RUN_NAME
CKPT=$OUT/ckpt
mkdir -p "$OUT"
echo "[run] $RUN_NAME N=$N iters=$TRAIN_ITERS warmup=$WARMUP_ITERS min_lr=$MIN_LR eval@$EVAL_INTERVAL" | tee "$OUT/run.meta"
echo "[run] shape=$SHAPE H=$H L=$L ffn=$FFN moe_ffn=$MOE_FFN gb=$GLOBAL_BATCH mb=$MICRO_BATCH lr=$PEAK_LR aux=$AUX_COEFF flops=$TARGET_FLOPS" | tee -a "$OUT/run.meta"

# ---- data (FLAME-style: weight 1.0 per tokenized .bin shard) ----
DATA_DIR=${DATA_DIR:-$ROOT/data/dclm_tokenized}
DATA_PATH=$(find "$DATA_DIR" -type f -name '*_text_document.bin' \
  -exec sh -c '[ -f "${1%.bin}.idx" ] && printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
if [ -z "$DATA_PATH" ]; then echo "ERROR: no tokenized part*_text_document.bin in $DATA_DIR"; exit 1; fi

export OMP_NUM_THREADS=16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=disabled
export HF_TOKEN=${HF_TOKEN:-}
# TE 1.11 runtime needs cudnn/cublas (pip nvidia-* packages) on the loader path
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export CUDNN_PATH=$NV/cudnn
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# Megatron compiles datasets/helpers_cpp via `make` calling bare python3/python3-config:
# put .venv first so pybind11 includes resolve.
export PATH=$ROOT/.venv/bin:$PATH

MODEL_ARGS=(
  --hidden-size $H --ffn-hidden-size $FFN --num-layers $L --num-attention-heads 16
  --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6
  --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear
  --moe-ffn-hidden-size $MOE_FFN --num-experts 64 --moe-router-topk 6
  --moe-shared-expert-intermediate-size $SHARED_INT --moe-layer-freq "$MOE_LAYER_FREQ"
  --moe-router-dtype fp32 --moe-router-pre-softmax --moe-router-score-function softmax
  --moe-aux-loss-coeff $AUX_COEFF --moe-z-loss-coeff 0.001
  --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model EleutherAI/pythia-12b
  # FLAME's native TransformerEngine path (faithful to FLAME). Single-GPU adaptations:
  #   --moe-grouped-gemm: batch the 64 local experts (EP=1) into one grouped GEMM
  #     (numerically equivalent to FLAME's EP=8 sequential-per-GPU experts; required for
  #     practical throughput with 64 experts on one GPU).
  #   --no-gradient-accumulation-fusion: that fusion needs apex (absent); perf-only, no math change.
  --transformer-impl transformer_engine --moe-grouped-gemm --no-gradient-accumulation-fusion
)
INFRA_ARGS=(
  --pipeline-model-parallel-size 1 --expert-model-parallel-size 1
  --use-distributed-optimizer --moe-token-dispatcher-type alltoall
  --distributed-timeout-minutes 30 --bf16
)
TRAIN_ARGS=(
  --micro-batch-size $MICRO_BATCH --global-batch-size $GLOBAL_BATCH
  --lr $PEAK_LR --min-lr $MIN_LR --lr-decay-style cosine
  --lr-warmup-iters $WARMUP_ITERS --lr-decay-iters $TRAIN_ITERS --train-iters $TRAIN_ITERS
  --weight-decay 0.01 --clip-grad 1.0 --seed $SEED
)
DATA_ARGS=( --seq-length 2048 --data-path $DATA_PATH --split 90,5,5 )
LOG_ARGS=(
  --log-interval 10 --log-throughput
  --save "$CKPT" --save-interval $SAVE_INTERVAL --load "$CKPT"
  --eval-interval $EVAL_INTERVAL --eval-iters 50
  --moe-per-layer-logging
)

cd Megatron-LM
if [ "${EVAL_ONLY:-0}" = "1" ]; then
  # criterion-4 per-expert load: load CKPT, skip training, eval-only with router hook
  export EXPERT_LOAD_OUT=$OUT/expert_load.json
  $ROOT/.venv/bin/torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29510} \
    $ROOT/scripts/phase0/expert_load.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" \
    --skip-train $EXTRA_ARGS \
    2>&1 | tee "$OUT/expert_load.log"
else
  $ROOT/.venv/bin/torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29510} pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" $EXTRA_ARGS \
    2>&1 | tee "$OUT/train.log"
fi
