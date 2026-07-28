#!/bin/bash
# Part A (orch 0141/0143): the two K=30 super-fine GRAIN=5 cells at STOCK SHARED_INT=352, run
# sequentially on the single H100. Cell = 320 experts / top-30 / routed moe_ffn 36 / shared 352,
# else identical to the panel (H256/9L, gb1024, mb32, 2121 iters, seed1234, WSD 3e-4->3e-5, div 2.9780).
# (1) temporal R=k=30 (native rolling residency, <=1 swap/token)  (2) full-MoE reference (free top-30).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
echo "[partA] $(date -u +%FT%TZ) START temporal g5"
GRAIN=5 MICRO_BATCH=32 TEMPORAL_EVICT=min_logit RUN_NAME=flame38m_g5_temporal RDZV_PORT=29551 \
  bash experiments/scale_1e18_1e19/flame38m_run.sh
rc1=$?
echo "[partA] $(date -u +%FT%TZ) temporal g5 exit=$rc1 ; START moe g5"
GRAIN=5 MOE_FULL=1 MICRO_BATCH=32 RUN_NAME=flame38m_g5_moe RDZV_PORT=29552 \
  bash experiments/scale_1e18_1e19/flame38m_run.sh
rc2=$?
echo "[partA] $(date -u +%FT%TZ) moe g5 exit=$rc2"
echo "[partA] BOTH K=30 RUNS DONE (temporal exit=$rc1, moe exit=$rc2)"
