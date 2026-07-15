#!/bin/bash
# H100 tail orchestrator: after the in-flight MoE@1e17 drive finishes (s1 running + s3), run the
# H100 temporal list (s2/s3@1e17 big shapes + sm1@1e16). A6000 handles the rest (see run plan doc).
set -uo pipefail
cd /workspace/FLAME-MoE
ROOT=$(pwd)
export TOKENIZER_MODEL=$ROOT/data/tok16k DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7600 GRAIN=3 MICRO_BATCH=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_AT_END=1 HF_TOKEN=
echo "=== H100 finish: waiting for MoE@1e17 (g3_moe_1e17.txt) to complete $(date) ==="
while pgrep -f "drive.sh experiments/isoflop_1e16_1e17/g3_moe_1e17.txt" >/dev/null; do sleep 30; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 5; done
echo "=== MoE@1e17 done, GPU free. Running H100 temporal (s2,s3@1e17 + sm1@1e16) $(date) ==="
export TEMPORAL=1 TEMPORAL_EVICT=min_logit
bash experiments/isoflop_1e16_1e17/drive.sh experiments/isoflop_1e16_1e17/g3_h100_tmoe.txt
echo "=== H100 ALL DONE $(date) ==="
