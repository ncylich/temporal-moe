#!/usr/bin/env bash
# Honest speed benchmark for the 1e19 fine-grained MoE: LR=0 so routing stays at init
# (balanced, all 192 experts busy -- the real regime; the earlier smokes skewed routing via
# a collapsed LR schedule and measured a fake 21.8 s/it). 8 iters, log every 2, profiler on
# iters 4-5. Config knobs via env: MB (default 16), GG (legacy|te), PF (1|0), EXTRA.
set -uo pipefail; cd /workspace/temporal-moe
. scripts/env.sh
export TOKENIZER_MODEL=EleutherAI/pythia-12b DATA_DIR=/root/data/dclm_tokenized
export PYTORCH_CUDA_ALLOC_CONF=${ALLOC:-expandable_segments:True} CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=4 HF_TOKEN=$(cat /root/.cache/huggingface/token)
NAME=${NAME:?}; MB=${MB:-16}; GG=${GG:-legacy}; PF=${PF:-1}; PROF=${PROF:-1}
GGARG=""; [ "$GG" = legacy ] && GGARG="--moe-use-legacy-grouped-gemm"
PARG=""; [ "$PROF" = 1 ] && PARG="--use-pytorch-profiler --profile --profile-step-start 4 --profile-step-end 5 --profile-ranks 0"
export EXTRA_ARGS="--no-rope-fusion $GGARG --train-iters ${ITERS:-8} --lr-decay-iters ${ITERS:-8} --lr-warmup-iters 1 --lr 0 --min-lr 0 --eval-iters 1 --eval-interval 1000 --save-interval 100000 --log-interval 2 $PARG ${EXTRA:-}"
echo "### bench $NAME mb=$MB gg=$GG pf=$PF prof=$PROF $(date -u +%H:%M)"; rm -rf results/phase0/runs/$NAME
env MOE_NO_LAYER_LOG=1 ${PF:+MOE_PERMUTE_FUSION=$PF} GRAIN=3 TEMPORAL=0 SHAPE=s19opt TARGET_FLOPS=1e19 PEAK_LR=3e-4 WARMUP_FRAC=0.01 LR_DECAY_STYLE=WSD \
  GLOBAL_BATCH=1024 MICRO_BATCH=$MB SEED=1234 RUN_NAME=$NAME timeout -k 30 ${BENCH_TIMEOUT:-1200} scripts/residency/gpu_lease.sh bash experiments/run.sh
grep -E " iteration +[0-9]+/" results/phase0/runs/$NAME/train.log | grep -oE "iteration +[0-9]+/|elapsed time per iteration \(ms\): [0-9.]+|load_balancing_loss: [0-9.E+-]+|TFLOP/s/GPU\): [0-9.]+" | paste - - - - | sed "s/^/[bench $NAME] /"
T=$(find results/phase0/runs/$NAME -name '*.pt.trace.json*' | head -1); [ -n "$T" ] && python3 /workspace/prof_summary.py "$T" | sed "s/^/[prof $NAME] /" && { [ "${KEEP:-0}" = 1 ] || rm -rf results/phase0/runs/$NAME/tb; }
echo "### bench $NAME DONE $(date -u +%H:%M)"
