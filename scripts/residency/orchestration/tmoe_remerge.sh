#!/usr/bin/env bash
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-rebuild-merged
A=/workspace/olmoe-adapt/data/gemma_ce_rebuild_adapter.pt
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 \
  --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### remerge DONE $(date -u +%H:%M)"
