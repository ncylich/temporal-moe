#!/bin/bash
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-fullpass-merged
A=/workspace/olmoe-adapt/data/gemma_ce_fullpass_adapter.pt
scripts/residency/disk_budget.sh || exit 3
if [ ! -d $M ]; then
  echo "### fullpass MERGE $(date -u +%H:%M)"
  CUDA_VISIBLE_DEVICES=0 /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
    --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 \
    --expert-lora-r 32 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true
fi
echo "### fullpass MERGE DONE $(date -u +%H:%M)"
/workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### fullpass VERIFY DONE $(date -u +%H:%M)"
GPUS=1,2,3 exec scripts/residency/grid_parallel.sh gemma $M gemma4_ce_fullpass
