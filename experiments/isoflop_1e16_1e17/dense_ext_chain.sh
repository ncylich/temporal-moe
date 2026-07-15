#!/bin/bash
# Dense left-arm extensions, 1e16 first. Common dense env; EVAL_AT_END differs per budget.
set -uo pipefail
cd "$(dirname "$0")/../.."
export DENSE=1 CE_FUSION=1 TOKENIZER_MODEL=/workspace/FLAME-MoE/data/tok16k DATA_DIR=/workspace/FLAME-MoE/data/tok16k_full BPB_DIVISOR=2.7568
echo "=== dense_sm1_1e16 (EVAL_AT_END) $(date) ==="
EVAL_AT_END=1 bash experiments/isoflop_1e16_1e17/drive.sh experiments/isoflop_1e16_1e17/dense_ext_1e16.txt
echo "=== dense_s1_1e17 (eval@iters/10) $(date) ==="
EVAL_AT_END=0 bash experiments/isoflop_1e16_1e17/drive.sh experiments/isoflop_1e16_1e17/dense_ext_1e17.txt
echo "=== DENSE EXT CHAIN DONE $(date) ==="
