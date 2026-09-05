#!/usr/bin/env bash
# Temporal-to-free curriculum arms at 1e17 (results/ablations/CURRICULUM_PLAN.md). Every arm is the
# recorded g<GRAIN>_moe_s2_1e17 configuration (shape s2, gb 256, mb 64, cosine lr 3e-3, 3,861 or
# 3,917 iterations, seed 1234) with the residency router installed and its constraint scheduled by
# iteration; residency changes no FLOPs, so every arm is compute-matched to the baseline. The 1e19
# fine-run speed recipe is on (bit-identical path). Final test eval is unconstrained in every arm.
#
# Arm names (schedules in fractions of the run; 'p' is the decimal point):
#   C0          full MoE through the router path (R = E): the environment control
#   SW<f>       R = k until fraction f, then free                       (plan C1 = SW0p5, C2 = SW0p667)
#   SWW<f>      SW<f> with a WSD lr schedule whose decay spans the free phase
#   RAMP<e>     R = k, 2k, 4k, 8k at 0, e/4, e/2, 3e/4, free from e     (plan C3 = RAMP0p75)
#   HET<a>-<b>  free fraction of sequences 0 until a, linear to 1 at b  (plan C4 = HET0p4-0p8)
#   SHD<l>      free routing, shadow resident set, coherence lambda l   (plan C5 = SHD0p01)
#   SAND        free, R = k over the third quarter, free                (plan C6)
#   WK<s>       weak constraint: s swaps per token at R = k, whole run; free re-score after (WK3 = k/2 on grain 1)
#   WK<s>SW<f>  weak constraint until fraction f, then free
#   WKA<s>-<f>  swaps annealed from s to k (free) at fraction f
#   tmoe_curriculum_1e17.sh [ARMS="SW0p5 C0 ..."] [GRAIN=3|1]
set -uo pipefail; cd "$(dirname "$0")/../../.."
. scripts/env.sh
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-EleutherAI/pythia-12b} DATA_DIR=${DATA_DIR:-/root/data/dclm_tokenized}   # SETUP=16k: the recorded 1e17 cells' tokenizer and corpus
[ "${SETUP:-}" = 16k ] && export TOKENIZER_MODEL=$PWD/data/tok16k DATA_DIR=$PWD/data/tok16k_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-5}
export EXTRA_ARGS="--no-rope-fusion --moe-use-legacy-grouped-gemm --cross-entropy-fusion-impl te"   # TE single-pass CE: 1.15 -> 0.85 s/it with mb 128, loss identical
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1 TEMPORAL_EVICT=min_logit   # the recorded temporal cells evict by min logit
export GRAIN=${GRAIN:-3} SHAPE=s2 TARGET_FLOPS=1e17 GLOBAL_BATCH=256 MICRO_BATCH=${MICRO_BATCH:-128} SEED=1234 TEMPORAL=1
if [ "$GRAIN" = 3 ]; then ITERS=3861; K=18; E=192; else ITERS=3917; K=6; E=64; fi
pct() { python3 -c "print(round($ITERS*$1))"; }
dec() { echo "${1/p/.}"; }
arm_env() {
  local a=${1%b} f          # a trailing b names a same-seed replicate of the arm (C0b, SHD0p01b)
  case $a in
    C0)    echo "TEMPORAL_RESIDENCY_R=$E" ;;                                              # C0b: same-seed replicate of C0 (run-to-run noise floor)
    SWW*)  f=$(dec ${a#SWW}); echo "TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $f):E LR_DECAY_STYLE=WSD WSD_DECAY_ITERS=$((ITERS - $(pct $f)))" ;;
    SW*)   f=$(dec ${a#SW});  echo "TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $f):E" ;;
    RAMP*) f=$(dec ${a#RAMP}); echo "TEMPORAL_ITER_SCHEDULE=0:$K,$(pct "$f/4"):$((2*K)),$(pct "$f/2"):$((4*K)),$(pct "3*$f/4"):$((8*K)),$(pct $f):E" ;;
    HET*)  f=${a#HET}; echo "TEMPORAL_FREE_FRAC_SCHEDULE=0:0,$(pct $(dec ${f%-*})):0,$(pct $(dec ${f#*-})):1 TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $(dec ${f#*-})):E" ;;   # the free fraction acts in training only; the iteration schedule makes evals free from the anneal's end
    SHD*)  f=$(dec ${a#SHD}); echo "TEMPORAL_SHADOW=1 TEMPORAL_COHERENCE_LAMBDA=$f" ;;
    SAND)  echo "TEMPORAL_ITER_SCHEDULE=0:E,$(pct 0.5):$K,$(pct 0.75):E" ;;
    WK*SW*) f=$(dec ${a#*SW}); local sw=${a#WK}; sw=${sw%%SW*}; echo "TEMPORAL_SWAPS=$sw TEMPORAL_ITER_SCHEDULE=0:$K,$(pct $f):E" ;;   # weak constraint (s swaps/token), free from f
    WKA*)  f=${a#WKA}; echo "TEMPORAL_SWAPS_SCHEDULE=0:${f%-*},$(pct $(dec ${f#*-})):$K" ;;                                     # weak constraint annealed to free (swaps = k) at fraction
    WK*)   echo "TEMPORAL_SWAPS=${a#WK}" ;;                                                                                    # weak constraint whole run, R = k; free re-score after
    *) echo "unknown arm $a" >&2; return 1 ;;
  esac
}
for A in ${ARMS:-SW0p5 C0 SW0p25 RAMP0p75 HET0p4-0p8 SHD0p01}; do   # SW0p667 dropped after SW0p5 lost 0.035 to C0: a later switch can only lose more
  NAME=cur_g${GRAIN}_1e17_$A${SETUP:+_$SETUP}; ENV=$(arm_env $A) || continue
  FINAL=results/phase0/runs/$NAME/ckpt/iter_$(printf %07d $ITERS)
  [ -d "$FINAL" ] && { echo "[skip] $NAME done"; continue; }
  echo "### curriculum $NAME [$ENV] START $(date -u +%H:%M)"
  env $ENV RUN_NAME=$NAME scripts/residency/gpu_lease.sh bash experiments/run.sh > /workspace/rerun-logs/cur_$NAME.out 2>&1
  echo "### curriculum $NAME rc=$? $(date -u +%H:%M)"
  "$PY" analysis/parse_run.py results/phase0/runs/$NAME 2>/dev/null | grep '^SUMMARY' | cut -c1-200
  case $A in WK[0-9]|WK[0-9]b)   # whole-run constrained: the logged final eval is constrained; re-score free (R = E) and native in place
    [ -d "$FINAL" ] && ! grep -q "^$NAME,cross," results/ablations/sweep_eval.csv 2>/dev/null && {
      cp results/phase0/runs/$NAME/run.meta results/phase0/runs/$NAME/run.meta.pre
      env $ENV SWEEPEVAL=1 RUN_NAME=$NAME TEMPORAL_RESIDENCY_R=$K SWEEP="native:$K cross:$E" scripts/residency/gpu_lease.sh timeout -k 60 1800 bash experiments/run.sh > /workspace/rerun-logs/cur_${NAME}_free.out 2>&1
      mv -f results/phase0/runs/$NAME/run.meta.pre results/phase0/runs/$NAME/run.meta
      grep -E "^\[sweep\]" /workspace/rerun-logs/cur_${NAME}_free.out | tail -2; } ;;
  esac
done
echo "### curriculum ALL DONE $(date -u +%H:%M)"
