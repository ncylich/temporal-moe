#!/bin/bash
# Single-GPU FLAME-MoE Phase-0 launcher (stock FLAME-MoE, B=1, EP=1, torchrun nproc=1).
# Adapted from scripts/training/flame-moe.sh: no SLURM, no GCS, local transformer impl.
# All knobs via env vars (see defaults). Computes train_iters so C = 6*N*D hits TARGET_FLOPS.
set -euo pipefail
# One environment contract: ROOT, PY, DATA_DIR, TOKENIZER_MODEL, CKPT_ROOT, NV.
. "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
cd "$ROOT"

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
  sm1) H=96;  L=4; FFN=512;  MOE_FFN=66;;
  s0)  H=128; L=4; FFN=684;  MOE_FFN=88;;
  s1) H=192; L=5;  FFN=1026; MOE_FFN=132;;
  s2) H=256; L=6;  FFN=1368; MOE_FFN=176;;
  s3) H=320; L=7;  FFN=1710; MOE_FFN=220;;
  s4) H=384; L=8;  FFN=2052; MOE_FFN=264;;
  s5) H=448; L=9;  FFN=2394; MOE_FFN=308;;
  s6) H=512; L=10; FFN=2736; MOE_FFN=352;;
  s19opt) H=800; L=14; FFN=4272; MOE_FFN=550;;  # 1e19 compute-optimal (t19 panel), N_active ~184.1M
  *) echo "bad SHAPE $SHAPE"; exit 1;;
esac
MOE_LAYER_FREQ="[0]+[1]*$((L-1))"
# s-dimension (constant/shared experts). s=1 (default): SHARED_MULT=2, TOPK=6 (FLAME stock).
# FLOP-matched s=2: SHARED_MULT=3, TOPK=5 -> active expert-FFN FLOPs identical (shared 3 + routed 5
# = 8 moe_ffn-units, same as s=1's shared 2 + routed 6); N and iters unchanged.
SHARED_MULT=${SHARED_MULT:-2}
SHARED_INT=$((SHARED_MULT * MOE_FFN))            # shared FFN sized from the ORIGINAL moe_ffn
# GRAIN>1: DeepSeek-style fine-graining of the ROUTED experts (compute-preserving). Subdivide each
# routed expert by GRAIN (moe_ffn/GRAIN, even-rounded for fused-SwiGLU), and scale the routed pool
# (num_experts) and top-k by GRAIN. Active routed FLOPs (topk*moe_ffn) stay ~fixed; the shared expert
# (SHARED_INT above) is NOT fine-grained. Temporal resident K = TOPK, still swaps 1 expert/token.
GRAIN=${GRAIN:-1}
NUM_EXPERTS=$((64 * GRAIN))
TOPK=${TOPK:-$((6 * GRAIN))}
if [ "$GRAIN" != "1" ]; then
  MOE_FFN=$("$PY" -c "print(2*round(($MOE_FFN/$GRAIN)/2))")
fi
# head_dim must be a multiple of 8 for TE fused attention. Fixed 16 heads gives head_dim
# 12/20/28 for s1/s3/s5 -> unfused slow path (~3x slower). Use heads=hidden/16 -> head_dim=16
# for every shape (identical params/FLOPs, so N and the law are unchanged; s2 already had 16).
NHEADS=$((H / 16))

# DENSE=1: vanilla dense transformer (no experts/router/shared) as the IsoFLOP floor. ffn_hidden is
# set so total non-embedding params == the MoE's ACTIVE non-embedding params at this scale, so the
# FLOP budget and iters carry over unchanged. (Even values: odd ffn crashes the fused-swiglu warmup.)
if [ "${DENSE:-0}" = "1" ]; then
  case $SHAPE in
    sm1) FFN=540;;  s0) FFN=716;;  s1) FFN=1068;; s2) FFN=1420;;
    s3) FFN=1772;;  s4) FFN=2124;; s5) FFN=2476;; s6) FFN=2828;;
    s19opt) FFN=4410;;
  esac
fi

# ---- compute iters so C = 6*N*D ----
read N TRAIN_ITERS < <("$PY" "$ROOT/analysis/shapes.py" iters "$SHAPE" "$TARGET_FLOPS" "$GLOBAL_BATCH")
WARMUP_ITERS=$("$PY" -c "print(max(1,round($WARMUP_FRAC*$TRAIN_ITERS)))")
MIN_LR=$("$PY" -c "print($PEAK_LR*0.1)")
# LR_DECAY_STYLE=WSD: flame-family schedule (stable then decay over the last ~10% of iters),
# for t18/t19 curve continuity. Default cosine == shipped behavior.
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
WSD_ARGS=""
[ "$LR_DECAY_STYLE" = "WSD" ] && WSD_ARGS="--lr-wsd-decay-iters ${WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
# Sweep runs (0c/0d) need eval@iters/10 to read the 1e16 point from the 1e17 run; HP runs only
# need the final val loss -> EVAL_AT_END=1 evaluates once at the end (saves ~9 intermediate evals).
if [ "${EVAL_AT_END:-0}" = "1" ]; then
  EVAL_INTERVAL=$TRAIN_ITERS
else
  EVAL_INTERVAL=$("$PY" -c "print(max(1,round($TRAIN_ITERS/10)))")   # 1e16 point = iters/10
fi
SAVE_INTERVAL=$EVAL_INTERVAL

OUT=$CKPT_ROOT/$RUN_NAME
CKPT=$OUT/ckpt
mkdir -p "$OUT"
echo "[run] $RUN_NAME N=$N iters=$TRAIN_ITERS warmup=$WARMUP_ITERS min_lr=$MIN_LR eval@$EVAL_INTERVAL" | tee "$OUT/run.meta"
echo "[run] shape=$SHAPE H=$H L=$L ffn=$FFN moe_ffn=$MOE_FFN grain=$GRAIN num_experts=$NUM_EXPERTS shared_int=$SHARED_INT dense=${DENSE:-0} temporal=${TEMPORAL:-0} shared_mult=$SHARED_MULT topk=$TOPK heads=$NHEADS gb=$GLOBAL_BATCH mb=$MICRO_BATCH lr=$PEAK_LR flops=$TARGET_FLOPS" | tee -a "$OUT/run.meta"

# ---- data (FLAME-style: weight 1.0 per tokenized .bin shard) ----
DATA_PATH=$(find "$DATA_DIR" -type f -name '*_text_document.bin' \
  -exec sh -c '[ -f "${1%.bin}.idx" ] && printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
if [ -z "$DATA_PATH" ]; then echo "ERROR: no tokenized part*_text_document.bin in $DATA_DIR"; exit 1; fi

export OMP_NUM_THREADS=16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=disabled
export HF_TOKEN=${HF_TOKEN:-}
# TE needs cudnn/cublas (pip nvidia-* packages) on the loader path, and Megatron compiles
# datasets/helpers_cpp via `make` calling bare python3/python3-config. Both the LD_LIBRARY_PATH
# and the PATH ordering are set by scripts/env.sh from the runtime-derived $NV.

# MoE-specific args (omitted entirely when DENSE=1 -> a plain dense SwiGLU transformer).
# Router scoring / aux-free paradigm (alignment program Track A):
#   default            : FLAME stock — pre-softmax softmax scoring + aux loss (unchanged behavior).
#   SIGMOID_SCORE=1    : sigmoid scoring, aux loss kept (the A0 attribution control).
#   AUXFREE=1          : DeepSeek-V3 aux-loss-free — sigmoid scoring (required by Megatron) +
#                        per-expert bias in SELECTION only, sign-updated per step
#                        (--moe-router-bias-update-rate, AUXFREE_U, default 1e-3). Callers should
#                        also set AUX_COEFF to the small backstop (e.g. 0.001) per the plan.
ROUTER_SCORE_ARGS="--moe-router-pre-softmax --moe-router-score-function softmax"
if [ "${AUXFREE:-0}" = "1" ]; then
  ROUTER_SCORE_ARGS="--moe-router-score-function sigmoid --moe-router-enable-expert-bias --moe-router-bias-update-rate ${AUXFREE_U:-1e-3}"
elif [ "${SIGMOID_SCORE:-0}" = "1" ]; then
  ROUTER_SCORE_ARGS="--moe-router-score-function sigmoid"
fi

MOE_ARGS=()
if [ "${DENSE:-0}" != "1" ]; then
  MOE_ARGS=(
    --moe-ffn-hidden-size $MOE_FFN --num-experts $NUM_EXPERTS --moe-router-topk $TOPK
    --moe-shared-expert-intermediate-size $SHARED_INT --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-router-dtype fp32 $ROUTER_SCORE_ARGS
    --moe-aux-loss-coeff $AUX_COEFF --moe-z-loss-coeff 0.001
    # --moe-grouped-gemm: batch the 64 local experts (EP=1) into one grouped GEMM (numerically
    # equivalent to FLAME's EP=8 sequential-per-GPU experts; required for throughput).
    --moe-grouped-gemm
    # MOE_PERMUTE_FUSION=1: fuse the token permute/unpermute (needs TE>=2.1). Cuts the
    # bandwidth-bound permute and avoids the fp32-router permute-memory blowup -> faster + less mem.
    ${MOE_PERMUTE_FUSION:+--moe-permute-fusion}
  )
fi

MODEL_ARGS=(
  --hidden-size $H --ffn-hidden-size $FFN --num-layers $L --num-attention-heads $NHEADS
  --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6
  --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear
  "${MOE_ARGS[@]}"
  --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model ${TOKENIZER_MODEL:-EleutherAI/pythia-12b}
  # FLAME's native TransformerEngine path; --no-gradient-accumulation-fusion needs no apex (perf-only).
  --transformer-impl transformer_engine --no-gradient-accumulation-fusion
  ${CE_FUSION:+--cross-entropy-loss-fusion}
)
INFRA_ARGS=(
  --pipeline-model-parallel-size 1 --expert-model-parallel-size 1
  --use-distributed-optimizer
  --distributed-timeout-minutes 30 --bf16
)
[ "${DENSE:-0}" != "1" ] && INFRA_ARGS+=(--moe-token-dispatcher-type alltoall)
TRAIN_ARGS=(
  --micro-batch-size $MICRO_BATCH --global-batch-size $GLOBAL_BATCH
  --lr $PEAK_LR --min-lr $MIN_LR --lr-decay-style $LR_DECAY_STYLE $WSD_ARGS
  --lr-warmup-iters $WARMUP_ITERS --lr-decay-iters $TRAIN_ITERS --train-iters $TRAIN_ITERS
  --weight-decay 0.01 --clip-grad 1.0 --seed $SEED
)
DATA_ARGS=( --seq-length 2048 --data-path $DATA_PATH --split 90,5,5 )
LOG_ARGS=(
  --log-interval 10 --log-throughput
  --save "$CKPT" --save-interval $SAVE_INTERVAL --load "$CKPT"
  --eval-interval $EVAL_INTERVAL --eval-iters 20
  --tensorboard-dir "$OUT/tb"     # sets a writer so track_moe_metrics logs each aux loss
)                                 # (load_balancing_loss, z_loss, coherence_loss) individually in
                                  # the train log + tensorboard, separate from (not summed into) 'lm loss'.
[ "${DENSE:-0}" != "1" ] && LOG_ARGS+=(--moe-per-layer-logging)

cd Megatron-LM
if [ "${PROBE:-0}" = "1" ]; then
  # Mechanistic router probe: load CKPT, log per-MoE-layer per-token routing on one fixed batch
  # (raw logits + resident mask). --finetune loads weights only; the hook records the first forward.
  export ROUTER_LOG_OUT=$OUT/router_log.pt
  "$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29510} \
    $ROOT/analysis/probes/router_probe.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" \
    --finetune --train-iters 6 --lr-warmup-iters 1 --save-interval 100000 --eval-iters 1 $EXTRA_ARGS \
    2>&1 | tee "$OUT/probe.log"
elif [ "${ACTPROBE:-0}" = "1" ]; then
  # Stability Part C: one forward pass, capture aggregate expert/trunk activation stats. Arch comes
  # from the parametrized MODEL_ARGS above, so this runs at any shape. TEMPORAL=1 makes
  # activation_probe.py install the residency router. Save to a throwaway dir so the --finetune
  # warmup iters never touch the real checkpoint; only the FIRST forward (uncorrupted weights) is recorded.
  export TEMPORAL=${TEMPORAL:-0} TEMPORAL_EVICT=${TEMPORAL_EVICT:-min_logit}
  export ACT_LOG_OUT=$OUT/act_log.pt
  "$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29510} \
    $ROOT/analysis/probes/activation_probe.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" \
    --finetune --train-iters 6 --lr-warmup-iters 1 --save-interval 100000 --eval-iters 1 \
    --save /tmp/probe_junk_ckpt $EXTRA_ARGS \
    2>&1 | tee "$OUT/actprobe.log"
elif [ "${QUANTEVAL:-0}" = "1" ]; then
  # Stability Part E: RTN fake-quant routed-expert weights (QUANT_BITS, group QUANT_GROUP) then
  # test-set eval. --finetune resets consumed_samples to 0 (end-of-training checkpoints have
  # consumed_samples ~= dataset size, so plain --skip-train's eval-print path misbehaves; this path
  # prints val+test cleanly). lr=0 freezes weights across the 2 warmup iters. fakequant_eval.py
  # quantizes only on the first EVAL-mode forward -> lands AFTER the last optimizer FP32-master->bf16
  # resync (which would undo it) and persists through eval. Save to throwaway; real checkpoint untouched.
  export TEMPORAL=${TEMPORAL:-0} TEMPORAL_EVICT=${TEMPORAL_EVICT:-min_logit}
  "$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29510} \
    $ROOT/analysis/probes/fakequant_eval.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" \
    --finetune --train-iters 2 --lr 0 --min-lr 0 --lr-warmup-iters 1 --save-interval 100000 \
    --save /tmp/probe_junk_ckpt --eval-iters ${EVAL_ITERS:-16} $EXTRA_ARGS \
    2>&1 | tee "$OUT/quanteval_b${QUANT_BITS}.log"
elif [ "${EVAL_ONLY:-0}" = "1" ]; then
  # criterion-4 per-expert load: load CKPT and run a few extra training iters so the router hook
  # fires on real forward passes of the trained model (--skip-train trips Megatron's val sampler).
  # The +3 iters at min-LR barely perturb the model.
  export EXPERT_LOAD_OUT=$OUT/expert_load.json
  # --finetune: load only model weights (reset iter/optim/sampler) -> run a few forward passes of the
  # trained model so the router hook records per-expert load. Tail args override the TRAIN/LOG arrays.
  # TEMPORAL=1: expert_load.py's forward patch never installs (and would override) the temporal
  # router, so a temporal checkpoint would SILENTLY eval as a full MoE. Route through
  # pretrain_temporal.py instead (installs the residency router + TEMPORAL_RHO/TEMPORAL_EMA_BETA
  # knobs; no load counting). EVAL_ITERS overrides eval length (default 1, unchanged).
  # NOTE: with --finetune the iteration counter resets, so the "warmup" train iters run at ~PEAK
  # LR (not min-LR as an earlier comment claimed) — 10 such iters measurably corrupt a checkpoint
  # before eval (smoke: BPB 1.93 vs 1.4753 baseline). The temporal eval path therefore freezes
  # weights outright (--lr 0 --min-lr 0): routers/banners still fire, eval is of the true ckpt.
  EVAL_ENTRY=$ROOT/analysis/probes/expert_load.py; EVAL_LOG=expert_load.log; EVAL_FREEZE=""
  if [ "${TEMPORAL:-0}" = "1" ]; then
    EVAL_ENTRY=$ROOT/temporal/pretrain_temporal.py; EVAL_LOG=eval_temporal.log
    EVAL_FREEZE="--lr 0 --min-lr 0"
  fi
  "$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29510} \
    $EVAL_ENTRY \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" \
    --finetune --train-iters 10 --lr-warmup-iters 1 --save-interval 100000 --eval-iters ${EVAL_ITERS:-1} $EVAL_FREEZE $EXTRA_ARGS \
    2>&1 | tee "$OUT/$EVAL_LOG"
else
  # TEMPORAL=1: rolling-residency MoE -> run via pretrain_temporal.py (installs the router patch,
  # then the identical pretrain loop). Same model args; only the expert selection differs.
  ENTRY=pretrain_gpt.py
  [ "${TEMPORAL:-0}" = "1" ] && ENTRY=$ROOT/temporal/pretrain_temporal.py
  "$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29510} $ENTRY \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}" $EXTRA_ARGS \
    2>&1 | tee "$OUT/train.log"
fi
