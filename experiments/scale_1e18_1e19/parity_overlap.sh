#!/bin/bash
# Parity test for the overlap-friendly variants (orch 0150 step 2). Runs g3-temporal 20 iters
# (mb32, seed1234, same data) four ways and prints the loss at iters 5/10/15/20:
#   (a) edited code, flags OFF        -> must equal (d) bit-for-bit (parity)
#   (b) edited code, V1 early-router  -> must DIFFER (flag active)
#   (c) edited code, V2 parallel-ffn  -> must DIFFER (flag active)
#   (d) ORIGINAL code (Megatron edits stashed), flags OFF -> the baseline
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
MEG=$ROOT/Megatron-LM
run20() {  # $1=run_name $2=port $3=extra_args
  rm -rf results/phase0/runs/"$1"
  GRAIN=3 MICRO_BATCH=32 TRAIN_ITERS=20 TEMPORAL_EVICT=min_logit \
    RUN_NAME="$1" RDZV_PORT="$2" EXTRA_MODEL_ARGS="$3" \
    bash experiments/scale_1e18_1e19/flame38m_run.sh > results/phase0/runs/_"$1".log 2>&1
  echo "== $1 (rc=$?) losses @ 5/10/15/20 =="
  grep -aE 'iteration +(5|10|15|20)/ *20' results/phase0/runs/"$1"/train.log 2>/dev/null | grep -aoE 'lm loss: [0-9.E+]+' || echo "  (no losses - see _$1.log)"
}
echo "### (a) edited OFF";            run20 parity_off  29561 ""
echo "### (b) edited V1 early-router"; run20 parity_v1   29562 "--overlap-early-router"
echo "### (c) edited V2 parallel-ffn"; run20 parity_v2   29563 "--overlap-parallel-ffn"
echo "### (d) ORIGINAL (stashed) OFF"
git -C "$MEG" stash push -m parity_overlap > /dev/null 2>&1 && STASHED=1 || STASHED=0
run20 parity_orig 29564 ""
[ "$STASHED" = 1 ] && git -C "$MEG" stash pop > /dev/null 2>&1
echo "### PARITY DONE (compare (a) vs (d): must match; (b),(c) must differ from (a))"
