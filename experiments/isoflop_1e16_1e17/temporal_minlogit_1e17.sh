#!/bin/bash
# Two temporal-MoE 1e17 cells: min_logit eviction, s2@1e17, 1-shared then 2-shared.
# Serial (one GPU); drive.sh skips any cell whose final checkpoint already exists.
# Export common env first: TOKENIZER_MODEL DATA_DIR CE_FUSION BPB_DIVISOR (absolute paths).
set -euo pipefail
cd "$(dirname "$0")/../.."
export TEMPORAL=1 EVAL_AT_END=1
cell() { echo "=== cell: evict=$1 shared_mult=$2 topk=$3 ($4) ==="; \
  TEMPORAL_EVICT="$1" SHARED_MULT="$2" TOPK="$3" bash experiments/isoflop_1e16_1e17/drive.sh "$4"; }
cell min_logit 2 6 experiments/isoflop_1e16_1e17/temporal_minlogit_sh1_1e17.txt   # 1 shared, K=6
cell min_logit 3 5 experiments/isoflop_1e16_1e17/temporal_minlogit_sh2_1e17.txt   # 2 shared, K=5
echo "=== MINLOGIT 1e17 PAIR DONE ==="
