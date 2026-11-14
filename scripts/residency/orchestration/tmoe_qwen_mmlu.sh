#!/usr/bin/env bash
# The one missing cell in the qwen table: MMLU. There is an adapted _dual run from before
# the reboot but no matched base, so it cannot be scored. Runs base and adapted through the
# SAME producer at the SAME arms, in sequence on the single GPU.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/root/models/qwen35-35b-a3b
M=/root/models/qwen35-rebuild-merged
A=/workspace/olmoe-adapt/data/qwen_ce_rebuild_adapter.pt

echo "### qwen base MMLU $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/mmlu_gptoss.py \
  --model qwen35_instruct --path $B --arms free,R8,R32 \
  --think off \
  --record-as qwen35_base_n_dual --gpu-mem 0.90

if [ ! -d "$M" ]; then
  echo "### merging qwen rebuild adapter $(date -u +%H:%M)"
  # adapter metadata: stack=hf+peft, expert_lora_r=16, family=qwen35 -> gemma trainer path
  /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
    --model $B --family qwen35 --no-unsloth --traj qwen35_d7_seq4096 --max-seq 4096 \
    --expert-lora-r 16 --opt adamw --micro-batch 16 --out $A --merge-out $M
  /workspace/venv_fla/bin/python analysis/residency/textify_qwen_merge.py $M
fi
echo "### qwen adapted MMLU $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/mmlu_gptoss.py \
  --model qwen35_instruct --path $M --arms free,R8,R32 \
  --think off \
  --record-as qwen35_ce_rebuild_n_dual --gpu-mem 0.90
echo "### qwen MMLU ALL DONE $(date -u +%H:%M)"
