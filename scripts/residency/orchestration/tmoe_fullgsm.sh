#!/bin/bash
# Full-split GSM8K (all 1,319 test problems, limit=0 -> lm_eval limit=None).
#
# WHY: at n=200 the paired McNemar SE on a single arm's R8-vs-free gap is ~2.2pt,
# so a cross-arm comparison carries SE ~3.0pt. Every adapted-arm difference measured
# so far (-0.5 to -9.5) sits inside that band, which is why six successive hypotheses
# each moved 1-3 points and none replicated. n=1319 cuts within-record SE to ~0.9 and
# cross-arm to ~1.2, making a 3-point effect a 2.5-sigma result instead of a 1-sigma one.
# No new data: this is the standard GSM8K test split, scored in full instead of sampled.
set -euo pipefail
cd /workspace/temporal-moe
REC=$1; MPATH=$2
export TMOE_ROOT=/workspace/temporal-moe HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
G=$(NEED_GB=100 TIMEOUT=14400 scripts/residency/wait_for_gpu.sh) || exit 1
export CUDA_VISIBLE_DEVICES=$G
echo "### fullgsm $REC on GPU $G $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path "$MPATH" --arms free,R8,R16 --record-as ${REC}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94
echo "### fullgsm $REC DONE $(date -u +%H:%M)"
