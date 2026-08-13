#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMOE_ROOT=/workspace/temporal-moe
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
export PATH=/opt/venv_vllm/bin:$PATH
cd /workspace/temporal-moe
PY=/opt/venv_vllm/bin/python
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-20b', local_dir='/dev/shm/gpt-oss-20b',
                  ignore_patterns=['original/*','*.pth','*.gguf','metal/*'])
PYEOF
echo "### HG2: 20b high gsm8k"
$PY -u analysis/residency/instruct_genbench_vllm.py --model gptoss_20b --arms free,R4 \
    --reasoning-effort high --record-as gptoss_20b_high --max-gen-toks 2048 \
    --backoff-cap 4096 --max-model-len 5632 --gpu-mem 0.85 \
    --tasks "gsm8k_cot_zeroshot=200" > /tmp/hg2_20b.log 2>&1
echo "### HG2: 20b OK"
rm -rf /dev/shm/gpt-oss-20b
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-120b', local_dir='/dev/shm/gpt-oss-120b',
                  ignore_patterns=['original/*','*.pth','*.gguf','metal/*'])
PYEOF
echo "### HG2: 120b high gsm8k"
$PY -u analysis/residency/instruct_genbench_vllm.py --model gptoss_120b --arms free,R4,R16 \
    --reasoning-effort high --record-as gptoss_120b_high --max-gen-toks 2048 \
    --backoff-cap 4096 --max-model-len 5632 --gpu-mem 0.92 \
    --tasks "gsm8k_cot_zeroshot=200" > /tmp/hg2_120b.log 2>&1
echo "### HG2: 120b OK"
rm -rf /dev/shm/gpt-oss-120b
git add results/ablations
git commit -q -m "gpt-oss high-effort gsm8k rerun at 4096 (filtered-length audit had masked 35% empty finals)"
git push -q origin layer-lexicality
echo "### HG2_DONE"
