#!/bin/bash
# Full temporal-MoE matrix: {lru, min_logit} x {1 shared, 2 shared} x {s0@1e16, s2@1e17} = 8 serial runs.
# One GPU -> strictly serial. drive.sh runs each regime's 1e16 point before its 1e17 point and skips any
# run whose final checkpoint already exists (idempotent: safe to re-launch / resume after a stop).
#
# Export the common env first (absolute paths, see docs/EVALUATION_METHODOLOGY.md s8f):
#   export TOKENIZER_MODEL=/workspace/FLAME-MoE/data/tok16k
#   export DATA_DIR=/workspace/FLAME-MoE/data/tok16k_full
#   export CE_FUSION=1 BPB_DIVISOR=2.7568
#
# RECOMMENDED: run the mechanism smoke alone and verify it BEFORE launching the full matrix
# (see docs/TEMPORAL_HANDOFF.md, Step 1). Once the smoke checkpoint exists this script skips it.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TEMPORAL=1
# Every cell is measured directly (1e16 and 1e17 are both explicit runs), so only the final BPB is
# needed -> EVAL_AT_END=1 evaluates once at the end (saves ~9 intermediate evals per run).
export EVAL_AT_END=1

regime() {  # evict shared_mult topk configfile
  echo "=== temporal regime: evict=$1 shared_mult=$2 topk=$3  ($4) ==="
  TEMPORAL_EVICT="$1" SHARED_MULT="$2" TOPK="$3" bash scripts/phase0/drive.sh "$4"
}

regime lru       2 6 scripts/phase0/temporal_lru_sh1.txt        # s0@1e16 line = mechanism smoke (runs first)
regime min_logit 2 6 scripts/phase0/temporal_minlogit_sh1.txt
regime lru       3 5 scripts/phase0/temporal_lru_sh2.txt
regime min_logit 3 5 scripts/phase0/temporal_minlogit_sh2.txt
