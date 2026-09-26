#!/bin/bash
# WritingBench for the rebuild arm -- the fifth cell. Base references already exist
# (gemma4_base_*: free 7.533, R8 7.460 pooled over the three disjoint 50-query subsets),
# so this makes the rebuild arm reportable on the full surface rather than 4 of 5.
set -euo pipefail
export GPU=${CUDA_VISIBLE_DEVICES:?slot must set the device}
exec /workspace/temporal-moe/scripts/residency/wb_arm.sh \
  /root/models/gemma4-rebuild-merged gemma4_rebuild R8,R16
