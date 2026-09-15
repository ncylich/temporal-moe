#!/usr/bin/env bash
# Re-measure the fused path at the configuration we ACTUALLY train at.
#
# The 22.7x was measured against a stock baseline of 93 tok/s (micro-batch 1, accum 16, seq 1024 --
# 8 tokens per expert through a Python loop over 128 experts). Stock in the real 50M training loop
# ran at 6,274 tok/s: 67x faster than the baseline the speedup was claimed against. The fused
# benchmark's own best was 6,046 tok/s, i.e. BELOW real stock -- but at seq 1024, so it is not
# like-for-like and cannot settle the question on its own.
#
# This runs the fused path at seq 2048 / mb 4, the config the 6,274 tok/s came from. Whatever it
# returns replaces the withdrawn claim.
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
for i in $(seq 1 120); do
  grep -qa "OLMOE LR SWEEP COMPLETE" /tmp/sweep_olmoe_chain.log 2>/dev/null && break
  sleep 60
done
echo "=== [$(date -u +%H:%M)] sweep done, re-measuring the fused claim ==="
/workspace/olmoe-adapt/venv25/bin/python bench_train_fused.py \
    --model /root/models/qwen3-30b-fused --seq 2048 --batches 2,4 --steps 4 \
    > /tmp/verify_fused.log 2>&1
echo "=== [$(date -u +%H:%M)] exit=$? ==="
echo "  real stock reference at seq 2048 mb4 accum2: 6,274 tok/s (from the 50M run)"
grep -aE "^ +[0-9]+ +[0-9]+ " /tmp/verify_fused.log | tail -3
echo "=== FUSED VERIFY COMPLETE ==="
