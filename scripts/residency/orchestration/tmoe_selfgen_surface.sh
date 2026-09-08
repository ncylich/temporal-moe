#!/usr/bin/env bash
# If selfgen is +5.3 on GSM8K but flat on IFEval/MMLU/code, the published gain is GSM8K
# overfitting and nothing else. Merge exists; evals only, each under the lease.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
M=/root/models/gemma4-selfgen-merged; L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
echo "### selfgen IFEval $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path $M --arms free,R8,R16 \
  --record-as gemma4_ce_selfgen_full --tasks "ifeval=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### selfgen MMLU $(date -u +%H:%M)"
$L $PY -u analysis/residency/mmlu_gptoss.py --model gemma4_instruct --path $M --arms free,R8,R16 \
  --record-as gemma4_ce_selfgen_full_dual --gpu-mem 0.90
echo "### selfgen HumanEval@8192 $(date -u +%H:%M)"
$L $PY -u analysis/residency/humaneval_gemma.py --path $M --arms free,R8,R16 --tag gemma4_ce_selfgen_he8192 \
  --max-tokens 8192 --max-model-len 9216
echo "### selfgen MBPP@8192 $(date -u +%H:%M)"
$L $PY -u analysis/residency/mbpp_gemma.py --path $M --arms free,R8,R16 --tag gemma4_ce_selfgen_m8192 \
  --max-tokens 8192 --max-model-len 9216 --gpu-mem 0.90
echo "### selfgen WritingBench $(date -u +%H:%M)"
export GPU=0; $L scripts/residency/wb_arm.sh $M gemma4_selfgen R8,R16
echo "### selfgen SURFACE DONE $(date -u +%H:%M)"
