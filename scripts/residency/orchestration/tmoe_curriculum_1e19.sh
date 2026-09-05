#!/usr/bin/env bash
# Promote a curriculum arm to 1e19: exactly the moe_fine_g3_1e19 configuration (tmoe_moe_fine_1e19.sh:
# s19opt, 192 experts k 18, gb 1024, mb 16, WSD lr 3e-4, 4,278 iterations, seed 1234) with the
# residency router installed and the arm's schedule rescaled to 4,278 iterations. Compared against
# moe_fine_g3_1e19 (test CE 3.1578). Auto-resume loop and checkpoint pruner as in the fine launcher.
# Expected ~5.3 days (~21 s/it) on one H100.   ARM=SW0p5 tmoe_curriculum_1e19.sh
set -uo pipefail; cd "$(dirname "$0")/../../.."
. scripts/env.sh
ARM=${ARM:?set ARM}; GRAIN=${GRAIN:-3}; if [ "$GRAIN" = 3 ]; then ITERS=4278; K=18; E=192; else ITERS=4318; K=6; E=64; fi   # recorded: moe_fine_g3_1e19 4278 iters, moe_coarse_1e19 4318 (gb 1024, lr 3e-4, WSD)
export TOKENIZER_MODEL=EleutherAI/pythia-12b DATA_DIR=/root/data/dclm_tokenized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-5}
export EXTRA_ARGS="--no-rope-fusion --moe-use-legacy-grouped-gemm --save-interval 200 --cross-entropy-fusion-impl te"
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
pct() { python3 -c "print(round($ITERS*$1))"; }
dec() { echo "${1/p/.}"; }
case $ARM in
  C0)    ENV="TEMPORAL_RESIDENCY_R=$E" ;;
  SWW*)  f=$(dec ${ARM#SWW}); ENV="TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $f):E WSD_DECAY_ITERS=$((ITERS - $(pct $f)))" ;;
  SW*)   f=$(dec ${ARM#SW});  ENV="TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $f):E" ;;
  RAMP*) f=$(dec ${ARM#RAMP}); ENV="TEMPORAL_ITER_SCHEDULE=0:$K,$(pct "$f/4"):$((2*K)),$(pct "$f/2"):$((4*K)),$(pct "3*$f/4"):$((8*K)),$(pct $f):E" ;;
  HET*)  f=${ARM#HET}; ENV="TEMPORAL_FREE_FRAC_SCHEDULE=0:0,$(pct $(dec ${f%-*})):0,$(pct $(dec ${f#*-})):1 TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $(dec ${f#*-})):E" ;;
  SHD*)  f=$(dec ${ARM#SHD}); ENV="TEMPORAL_SHADOW=1 TEMPORAL_COHERENCE_LAMBDA=$f" ;;
  SAND)  ENV="TEMPORAL_ITER_SCHEDULE=0:E,$(pct 0.5):$K,$(pct 0.75):E" ;;
  WK*)   ENV="TEMPORAL_SWAPS=${ARM#WK}" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
NAME=cur_g${GRAIN}_1e19_$ARM
FINAL=results/phase0/runs/$NAME/ckpt/iter_$(printf %07d $ITERS)
( while true; do ls -d results/phase0/runs/$NAME/ckpt/iter_* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf; sleep 300; done ) &
PRUNER=$!; trap 'kill $PRUNER 2>/dev/null' EXIT
echo "### curriculum1e19 $NAME [$ENV] START $(date -u +%H:%M)"
for attempt in 1 2 3 4 5 6; do
  [ -d "$FINAL" ] && break
  echo "### curriculum1e19 $NAME attempt $attempt $(date -u +%H:%M)"
  env $ENV MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1 TEMPORAL_EVICT=min_logit GRAIN=$GRAIN TEMPORAL=1 SHAPE=s19opt TARGET_FLOPS=1e19 PEAK_LR=3e-4 WARMUP_FRAC=0.01 LR_DECAY_STYLE=WSD GLOBAL_BATCH=1024 MICRO_BATCH=16 SEED=1234 \
    RUN_NAME=$NAME scripts/residency/gpu_lease.sh bash experiments/run.sh > /workspace/rerun-logs/cur_$NAME.attempt$attempt.out 2>&1
  echo "### curriculum1e19 $NAME attempt $attempt rc=$? $(date -u +%H:%M)"
  [ -d "$FINAL" ] || sleep 120
done
"$PY" analysis/parse_run.py results/phase0/runs/$NAME 2>/dev/null | grep '^SUMMARY'
echo "### curriculum1e19 $NAME DONE $(date -u +%H:%M)"
