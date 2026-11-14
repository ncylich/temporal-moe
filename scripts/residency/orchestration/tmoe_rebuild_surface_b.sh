#!/bin/bash
# Lane B: HumanEval via the channel-aware producer (gemma4 emits <channel|>, so the
# stock humaneval_instruct scores every cell 0.000), then re-measure the short640 length
# arm on the FULL GSM8K split -- its n=200 falsification is void now that the base gap is
# known to be -9.0 rather than -6.0.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_vllm312/bin/python
echo "### rebuild HumanEval (channel-aware) $(date -u +%H:%M)"
$PY -u analysis/residency/humaneval_gemma.py \
  --path /root/models/gemma4-rebuild-merged \
  --arms free,R8,R16 --tag gemma4_ce_rebuild_full
echo "### short1pass GSM8K n=1319 $(date -u +%H:%M)"
$PY -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path /root/models/gemma4-short1pass-merged \
  --arms free,R8,R16 --record-as gemma4_ce_short1pass_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### rebuild LANE-B DONE $(date -u +%H:%M)"
