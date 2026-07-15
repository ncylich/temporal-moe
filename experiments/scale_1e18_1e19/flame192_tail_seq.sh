#!/bin/bash
# 1e18 LEFT-FLANK TAIL on H100: offload the a6000's rear runs so the h192 left flank finishes sooner.
# Tail-first order (g3_temporal -> g3_moe -> g1_temporal) so the a6000-adjacent cell (g1_temporal) runs
# LAST, minimizing overlap risk with the a6000 (which is cut after g1_moe). h192 config identical to
# flame192_seq.sh (same 50k dclm / CE / WSD / seed 1234 -> CE comparable across boxes). Restart-safe skip.
set -uo pipefail
cd "$(dirname "$0")/../.."
C="HIDDEN_SIZE=192 N_LAYERS=9 N_HEADS=12 FFN_MOE=1026 FFN_DENSE=1056 MOE_FFN_BASE=132 SHARED_INT=264 TRAIN_ITERS=3068 MICRO_BATCH=32"
run(){ local name=$1; shift
  if [ -f "results/phase0/runs/$name/ckpt/latest_checkpointed_iteration.txt" ]; then echo "SKIP $name (done)"; return; fi
  echo "--- RUN $name $(date) ---"; env $C "$@" RUN_NAME=$name bash experiments/scale_1e18_1e19/flame_scale_run.sh; }
echo "=== flame192 left-flank TAIL (H100) SEQ START $(date) ==="
run flame192_g3_temporal  GRAIN=3 TEMPORAL_EVICT=min_logit       RDZV_PORT=29594
run flame192_g3_moe       MOE_FULL=1 GRAIN=3                     RDZV_PORT=29592
run flame192_g1_temporal  GRAIN=1 TEMPORAL_EVICT=min_logit       RDZV_PORT=29593
echo "=== flame192 TAIL SEQ DONE $(date) ==="
