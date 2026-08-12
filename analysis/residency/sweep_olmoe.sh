#!/usr/bin/env bash
# Stage 1: OLMoE LR sweep. Cheapest model, most constraint damage, matched null already on disk.
# 15M tokens, evals at 5M/10M/15M. Selection on raw constrained BPB -- no null needed to rank.
#
# --lora 32 --lora-attn 32: EXPERT LoRA plus attention LoRA, matching every published OLMoE cell
# (ce_auxfix_free_attn_50M, ce_auxfix_50M, ce_freeall_50M all carry lora=32). The first version
# of this sweep used --lora 0, adapting router + norms + attention while leaving all 64 experts
# frozen -- ~90% of the model, and the component residency actually reroutes tokens between.
# That also made last night's ce_attn_nofree_50M non-comparable to the null it was scored
# against (ce_freeall_50M carries expert LoRA and no attention LoRA), which invalidates its
# 26.0% recovery figure.
#
# mb=4 at seq 4096 = 16384 tokens/step, matching the Qwen sweeps. The 50M OLMoE cell used mb=16
# (65536 tok/step); sweeping there and transferring to Qwen at 16384 would move the optimum by
# batch size (~sqrt(batch) for Adam) rather than by model. Cross-model transfer is the point of
# sweeping the cheap model, so batch is held equal and OLMoE's own 50M baseline is re-derived at
# the winning LR instead.
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
PY=/workspace/olmoe-adapt/venv/bin/python
for LR in 1e-5 3e-5 1e-4 3e-4 1e-3; do
  echo "=== [$(date -u +%H:%M)] olmoe lr=$LR ==="
  $PY train_ple.py --tag "sweep_lr${LR}" --rank off --lora 32 --lora-attn 32 --tokens 15000000 \
      --lr "$LR" --mb 4 --accum 1 --eval-every 5000000 > "/tmp/sweep_olmoe_lr${LR}.log" 2>&1
  echo "=== [$(date -u +%H:%M)] lr=$LR exit=$? ==="
done
echo "=== OLMOE LR SWEEP COMPLETE ==="
