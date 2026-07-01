#!/bin/bash
# Overnight: after the running min_logit 2-vs-1-shared 1e17 pair finishes, sweep the 5 remaining
# temporal min_logit 1-shared points over the dense-matched shapes. Serial; idempotent (drive skips done).
set -uo pipefail
cd /workspace/FLAME-MoE
while pgrep -f temporal_minlogit_1e17.sh >/dev/null; do sleep 30; done   # wait for sh2 pair
export TOKENIZER_MODEL=/workspace/FLAME-MoE/data/tok16k DATA_DIR=/workspace/FLAME-MoE/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7568
export TEMPORAL=1 TEMPORAL_EVICT=min_logit SHARED_MULT=2 TOPK=6 EVAL_AT_END=1
echo "=== sweep start $(date) ==="
bash scripts/phase0/drive.sh scripts/phase0/temporal_minlogit_sh1_sweep.txt
echo "=== TEMPORAL MINLOGIT SH1 SWEEP DONE $(date) ==="
