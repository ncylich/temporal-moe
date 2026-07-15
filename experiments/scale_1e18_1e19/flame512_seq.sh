#!/bin/bash
# 1e18 RIGHT-FLANK (FLAME-98M rung, hidden-512): 5 configs, serial, H100.
# h512/L9/32heads, ffn 2736(moe)/2826(dense), moe_ffn_base 352, shared 704, ~99M active (incl embed),
# 802 iters=1e18, mb32. ffn_dense param-matched to MoE active (same method -> 38M's 1422; anchored 38M=2121).
# Skips any run whose final checkpoint already exists (restart-safe). Shared 50k dclm / CE / WSD, seed 1234.
set -uo pipefail
cd "$(dirname "$0")/../.."
C="HIDDEN_SIZE=512 N_LAYERS=9 N_HEADS=32 FFN_MOE=2736 FFN_DENSE=2826 MOE_FFN_BASE=352 SHARED_INT=704 TRAIN_ITERS=802 MICRO_BATCH=32"
run(){ local name=$1; shift
  if [ -f "results/phase0/runs/$name/ckpt/latest_checkpointed_iteration.txt" ]; then echo "SKIP $name (done)"; return; fi
  echo "--- RUN $name $(date) ---"; env $C "$@" RUN_NAME=$name bash experiments/scale_1e18_1e19/flame_scale_run.sh; }
echo "=== flame512 1e18 right-flank SEQ START $(date) ==="
run flame512_dense        DENSE=1                                RDZV_PORT=29580
run flame512_g1_moe       MOE_FULL=1 GRAIN=1                     RDZV_PORT=29581
run flame512_g1_temporal  GRAIN=1 TEMPORAL_EVICT=min_logit       RDZV_PORT=29583
run flame512_g3_moe       MOE_FULL=1 GRAIN=3                     RDZV_PORT=29582
run flame512_g3_temporal  GRAIN=3 TEMPORAL_EVICT=min_logit       RDZV_PORT=29584
echo "=== flame512 SEQ DONE $(date) ==="
