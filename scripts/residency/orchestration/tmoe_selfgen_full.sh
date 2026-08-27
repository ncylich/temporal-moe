#!/usr/bin/env bash
# HYPOTHESIS: D12's +6.0 came from its math_selfgen lane (model-generated, GSM8K-shaped
# word problems), which the rebuild replaced with StackMathQA. On the identical 200
# questions: D12 +6.0, selfgen +3.0, rebuild +0.5. If selfgen reaches ~+5 at n=1319 where
# the rebuild reaches +2.3, the gap to D12 is style-matching, not lost data.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-selfgen-merged
A=/workspace/olmoe-adapt/data/gemma_ce_selfgen_adapter.pt; L=scripts/residency/gpu_lease.sh
echo "### selfgen merge (traj=gemma4_d7_seq4096 r=32) $(date -u +%H:%M)"
[ -d $M ] || { $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 \
  --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
/workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### selfgen eval $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_selfgen_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### selfgen n1319 DONE $(date -u +%H:%M)"
