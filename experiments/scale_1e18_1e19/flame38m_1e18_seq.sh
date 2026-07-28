#!/bin/bash
# 1e18 final sequence (local self-consistent 50k/CE panel). mb=32 all runs (probed to fit).
# Order: user's 3 (G3-temporal, G3-MoE, G1-MoE) then the dense-floor reference.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
MB=32
echo "=== 1e18 SEQ START $(date) ==="
echo "--- [1/4] G3-temporal ---"
GRAIN=3 MICRO_BATCH=$MB TEMPORAL_EVICT=min_logit RUN_NAME=flame38m_g3_temporal RDZV_PORT=29540 bash experiments/scale_1e18_1e19/flame38m_run.sh
echo "--- [2/4] G3-MoE ---"
GRAIN=3 MOE_FULL=1 MICRO_BATCH=$MB RUN_NAME=flame38m_g3_moe RDZV_PORT=29541 bash experiments/scale_1e18_1e19/flame38m_run.sh
echo "--- [3/4] G1-MoE (measured MoE control) ---"
GRAIN=1 MOE_FULL=1 MICRO_BATCH=$MB RUN_NAME=flame38m_g1_moe RDZV_PORT=29542 bash experiments/scale_1e18_1e19/flame38m_run.sh
echo "--- [4/4] dense floor (reference) ---"
DENSE=1 MICRO_BATCH=$MB RUN_NAME=flame38m_dense RDZV_PORT=29543 bash experiments/scale_1e18_1e19/flame38m_run.sh
echo "=== 1e18 SEQ DONE $(date) ==="
