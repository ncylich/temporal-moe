#!/bin/bash
# V1 early-router 1e18-g3 promotion cells (orch 0150/0160). Exact flame38m 1e18 config (GRAIN=3=192exp/
# top18/moe_ffn58/shared352, H256/L9, gb1024, mb32, 2121 iters, seed1234, WSD, pythia-50k, divisor 2.9780)
# + --overlap-early-router via EXTRA_MODEL_ARGS. temporal + moe reference, sequential on one H100.
# Compare vs standard 1e18-g3: temporal 1.3354, moe 1.3461 (test BPB @ 2.9780).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
echo "[v1_1e18] $(date -u +%FT%TZ) START temporal"
GRAIN=3 MICRO_BATCH=32 TEMPORAL_EVICT=min_logit EXTRA_MODEL_ARGS="--overlap-early-router" \
  RUN_NAME=flame38m_g3_temporal_ovlEarly RDZV_PORT=29611 \
  bash experiments/scale_1e18_1e19/flame38m_run.sh
echo "[v1_1e18] $(date -u +%FT%TZ) temporal rc=$? ; START moe"
GRAIN=3 MOE_FULL=1 MICRO_BATCH=32 EXTRA_MODEL_ARGS="--overlap-early-router" \
  RUN_NAME=flame38m_g3_moe_ovlEarly RDZV_PORT=29612 \
  bash experiments/scale_1e18_1e19/flame38m_run.sh
echo "[v1_1e18] $(date -u +%FT%TZ) moe rc=$?"
echo "[v1_1e18] BOTH V1 1e18-g3 CELLS DONE"
