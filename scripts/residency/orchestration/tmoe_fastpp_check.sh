#!/usr/bin/env bash
# Fast presence penalty vs vLLM native: same engine/config/seed, batch 256, temp 0.7 / top-p 0.8 / pp 1.5. Expect identical generations and ~2x tok/s.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
S=/tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad; L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
run() { name=$1; shift; echo "### fastpp $name $(date -u +%H:%M)"; "$@" $L $PY -u analysis/residency/parity_vllm.py --path /root/models/qwen35-35b-a3b --adapter /workspace/olmoe-adapt/data/qwen_ce_digit3_adapter.pt --R 8 --n 256 --max-new 1024 --think off --gpu-mem 0.65 --max-model-len 2560 --max-num-seqs 256 --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 --out $S/fastpp_$name.json > /workspace/rerun-logs/fastpp_$name.log 2>&1; echo "### $name rc=$?"; grep -E "^\[parity\] (free|R8)|Traceback|Error" /workspace/rerun-logs/fastpp_$name.log | tail -3; }
run native env TEMPORAL_FAST_PP=0
run fast env TEMPORAL_FAST_PP=1
$PY analysis/residency/parity_vllm.py --compare $S/fastpp_native.json $S/fastpp_fast.json
echo "### fastpp DONE $(date -u +%H:%M)"
