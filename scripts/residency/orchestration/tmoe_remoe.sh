#!/usr/bin/env bash
# BASELINE #2 (ReMoE, Zhu et al.) -- faithful remake per BASELINE_METHODS_COMPARISON.md:
# router-only finetune, recency-bias reuse objective, residency constraint OFF during
# training. Their recipe is ~33M tokens; we use the same 3.4M budget as every other arm
# here so the comparison is against our own ladder rather than their compute.
# Evaluate free and R8: their method never bounds the resident set, so the question is
# whether better reuse alone buys constrained quality.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/dev/shm/gemma4-26b-it
A=/workspace/olmoe-adapt/data/gemma_remoe_adapter.pt
M=/root/models/gemma4-remoe-merged
COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096
        --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A"
scripts/residency/wait_gpu_free.sh 120 21600
echo "### remoe train $(date -u +%H:%M)"
[ -s "$A" ] || /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py \
  $COMMON --router-only --no-constraint --remoe-lambda 1.0 --remoe-gamma 0.9 \
  --accum 16 --lr 3e-5 --tokens 3400000
scripts/residency/wait_gpu_free.sh 120 3600
echo "### remoe merge $(date -u +%H:%M)"
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  $COMMON --merge-out $M; cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M || true
scripts/residency/wait_gpu_free.sh 120 3600
echo "### remoe eval $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8 --record-as gemma4_remoe_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### remoe ALL DONE $(date -u +%H:%M)"
