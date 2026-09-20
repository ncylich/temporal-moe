#!/bin/bash
# qwen3.5 full-split GSM8K, think-off, arms free/R8/R32 (E=256, k=8 -> 3.1% and 12.5%).
# The committed qwen rows are not comparable: the base was run at free,R8,R32 but
# qwen35_ce_rebuild at free,R8,R16, so no adapted-vs-base gap was ever valid for this
# model. Both sides are re-run here on the same arms at n=1319. Sampling matches the
# non-thinking card recipe used for the committed think-off cells.
#   tmoe_qwen_full.sh <record> <model-path> [adapter-suffix]
set -euo pipefail
cd /workspace/temporal-moe
REC=$1; MPATH=$2; SUF=${3:-}
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "$SUF" ] && [ ! -d "$MPATH" ]; then
  scripts/residency/disk_budget.sh || exit 3
  A=/workspace/olmoe-adapt/data/qwen_ce_${SUF}_adapter.pt
  [ -s "$A" ] || { echo "### no adapter $A"; exit 4; }
  # The adapter's own metadata says stack=hf+peft, expert_lora_r=16, family=qwen35 --
  # it was produced by train_gemma_ce.py --family qwen35 --no-unsloth, NOT by
  # train_qwen_ce.py (which is the unsloth stack and names its tensors differently;
  # merging with it fails on 411 unmatched keys). Rank must match the checkpoint.
  /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
    --model /root/models/qwen35-35b-a3b --family qwen35 --no-unsloth \
    --traj qwen35_d7_seq4096 --max-seq 4096 --expert-lora-r 16 \
    --opt adamw --micro-batch 16 --out $A --merge-out $MPATH
  scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py \
    --base /root/models/qwen35-35b-a3b --merged $MPATH
fi
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model qwen35_instruct --path "$MPATH" --arms free,R8,R32 --record-as ${REC}_n1319 \
  --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
echo "### qwen ${REC} n1319 DONE $(date -u +%H:%M)"
