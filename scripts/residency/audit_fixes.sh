#!/bin/bash
# Audit remediation: (1) qwen free gsm8k re-run below the cutover (kills the
# authoritative-vs-never-cite contradiction); (2) LFM R4 ifeval at full 541 items
# (matches the free arm's item set).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMOE_ROOT=/workspace/temporal-moe
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
export PATH=/opt/venv_vllm/bin:$PATH
cd /workspace/temporal-moe
PY=/opt/venv_vllm/bin/python
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-35B-A3B', local_dir='/dev/shm/qwen35-35b-a3b-instruct',
                  ignore_patterns=['original/*','*.pth','*.gguf'])
print('qwen staged')
PYEOF
echo "### AF: qwen free gsm8k"
$PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct \
    --path /dev/shm/qwen35-35b-a3b-instruct --arms free --presence-penalty 1.5 \
    --max-gen-toks 2048 --backoff-cap 4096 --max-model-len 7168 --gpu-mem 0.94 \
    --tasks "gsm8k_cot_zeroshot=0" > /tmp/af_qwen.log 2>&1
echo "### AF: qwen OK"
rm -rf /dev/shm/qwen35-35b-a3b-instruct
echo "### AF: lfm R4 ifeval full"
$PY -u analysis/residency/instruct_genbench_vllm.py --model lfm25_instruct \
    --arms R4 --max-gen-toks 1024 --backoff-cap 8192 --max-model-len 9216 \
    --tasks "ifeval=0" > /tmp/af_lfm.log 2>&1
echo "### AF: lfm OK"
git add results/ablations
git commit -q -m "Audit fixes: qwen free gsm8k below cutover; LFM R4 ifeval at full item set"
git push -q origin layer-lexicality
echo "### AF_DONE"
