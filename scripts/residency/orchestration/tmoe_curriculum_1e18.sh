#!/usr/bin/env bash
# Promote a 1e17 curriculum arm to 1e18: the flame38m_g3 configuration (experiments/scale_1e18_1e19/
# flame38m_run.sh temporal mode: 9 layers, gb 1024, mb 32, WSD lr 3e-4, 2,121 iterations, seed 1234)
# with the arm's schedule rescaled to 2,121 iterations. Compared against the flame38m_g3_moe seed
# triplet (4.0087, 4.0170, 4.0152; mean 4.0136). Speed recipe on. ~7.5 h.
#   ARM=SW0p5 tmoe_curriculum_1e18.sh
set -uo pipefail; cd "$(dirname "$0")/../../.."
. scripts/env.sh
ARM=${ARM:?set ARM}; ITERS=2121; K=18; E=192
export CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-5} HF_TOKEN=$(cat /root/.cache/huggingface/token)
export MOE_TORCH_GMM=1 MOE_NO_LAYER_LOG=1 EXTRA_ARGS="--moe-permute-fusion --no-rope-fusion --moe-use-legacy-grouped-gemm"
pct() { python3 -c "print(round($ITERS*$1))"; }
dec() { echo "${1/p/.}"; }
case $ARM in
  C0)    ENV="TEMPORAL_RESIDENCY_R=$E" ;;
  SWW*)  f=$(dec ${ARM#SWW}); ENV="TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $f):E WSD_DECAY_OVERRIDE=$((ITERS - $(pct $f)))" ;;
  SW*)   f=$(dec ${ARM#SW});  ENV="TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $f):E" ;;
  RAMP*) f=$(dec ${ARM#RAMP}); ENV="TEMPORAL_ITER_SCHEDULE=0:$K,$(pct "$f/4"):$((2*K)),$(pct "$f/2"):$((4*K)),$(pct "3*$f/4"):$((8*K)),$(pct $f):E" ;;
  HET*)  f=${ARM#HET}; ENV="TEMPORAL_FREE_FRAC_SCHEDULE=0:0,$(pct $(dec ${f%-*})):0,$(pct $(dec ${f#*-})):1 TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $(dec ${f#*-})):E" ;;
  SHD*)  f=$(dec ${ARM#SHD}); ENV="TEMPORAL_SHADOW=1 TEMPORAL_COHERENCE_LAMBDA=$f" ;;
  SAND)  ENV="TEMPORAL_ITER_SCHEDULE=0:E,$(pct 0.5):$K,$(pct 0.75):E" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
NAME=cur_flame38m_g3_$ARM
FINAL=results/phase0/runs/$NAME/ckpt/iter_$(printf %07d $ITERS)
[ -d "$FINAL" ] && { echo "[skip] $NAME done"; exit 0; }
echo "### curriculum1e18 $NAME [$ENV] START $(date -u +%H:%M)"
env $ENV GRAIN=3 MICRO_BATCH=32 TEMPORAL_EVICT=min_logit RUN_NAME=$NAME RDZV_PORT=29640 \
  scripts/residency/gpu_lease.sh bash experiments/scale_1e18_1e19/flame38m_run.sh > /workspace/rerun-logs/cur_$NAME.out 2>&1
echo "### curriculum1e18 $NAME rc=$? $(date -u +%H:%M)"
grep "on test set" results/phase0/runs/$NAME/train.log 2>/dev/null | tail -1 | cut -c1-160
