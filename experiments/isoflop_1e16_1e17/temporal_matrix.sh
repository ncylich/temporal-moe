#!/bin/bash
# Full temporal-MoE matrix: {lru, min_logit} x {1 shared, 2 shared} x {s0@1e16, s2@1e17} = 8 serial runs.
# Cheap-first: ALL FOUR 1e16 cells run before any 1e17 cell, so a problem surfaces on the cheap runs.
# One GPU -> strictly serial. drive.sh skips any run whose final checkpoint already exists (idempotent:
# safe to re-launch / resume, and it skips the smoke cell if you ran it first per the handoff).
#
# Export the common env first (absolute paths, see docs/EVALUATION_METHODOLOGY.md s8f):
#   export TOKENIZER_MODEL=$ROOT/data/tok16k
#   export DATA_DIR=$ROOT/data/tok16k_full
#   export CE_FUSION=1 BPB_DIVISOR=2.7568
#
# RECOMMENDED: run the mechanism smoke alone and verify it BEFORE launching this (handoff Step 1).
set -euo pipefail
# One environment contract: ROOT, PY, DATA_DIR, TOKENIZER_MODEL, CKPT_ROOT, NV.
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
export TEMPORAL=1
# Every cell is measured directly, so only the final BPB is needed -> EVAL_AT_END=1 (saves ~9 evals/run).
export EVAL_AT_END=1

cell() {  # evict shared_mult topk configfile
  echo "=== temporal cell: evict=$1 shared_mult=$2 topk=$3  ($4) ==="
  TEMPORAL_EVICT="$1" SHARED_MULT="$2" TOPK="$3" bash experiments/isoflop_1e16_1e17/drive.sh "$4"
}

# --- all 1e16 first (cheap; the first cell is also the mechanism smoke) ---
cell lru       2 6 experiments/isoflop_1e16_1e17/temporal_lru_sh1_1e16.txt        # = the smoke cell
cell min_logit 2 6 experiments/isoflop_1e16_1e17/temporal_minlogit_sh1_1e16.txt
cell lru       3 5 experiments/isoflop_1e16_1e17/temporal_lru_sh2_1e16.txt
cell min_logit 3 5 experiments/isoflop_1e16_1e17/temporal_minlogit_sh2_1e16.txt

# --- then all 1e17 ---
cell lru       2 6 experiments/isoflop_1e16_1e17/temporal_lru_sh1_1e17.txt
cell min_logit 2 6 experiments/isoflop_1e16_1e17/temporal_minlogit_sh1_1e17.txt
cell lru       3 5 experiments/isoflop_1e16_1e17/temporal_lru_sh2_1e17.txt
cell min_logit 3 5 experiments/isoflop_1e16_1e17/temporal_minlogit_sh2_1e17.txt
