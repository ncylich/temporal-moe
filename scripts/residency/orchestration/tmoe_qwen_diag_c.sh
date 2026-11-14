#!/usr/bin/env bash
# stage C: which engine config reproduces the in-process drift? standalone merged checkpoint with
# C1 max_model_len 2560, C2 sleep/wake cycle, C3 both + no warm-up call; each vs the 0.85 reference
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TEMPORAL_EAGER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
S=/tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad; L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; M=/root/models/qwen35-digit3-merged
run() { name=$1; shift; echo "### qwen diag C $name $(date -u +%H:%M)"; $L $PY -u analysis/residency/parity_vllm.py --path $M --R 8 --n 8 --max-new 256 --think off --gpu-mem 0.55 --out $S/ref_qwen_$name.json "$@" && $PY analysis/residency/parity_vllm.py --compare $S/ref_qwen_digit3_eager.json $S/ref_qwen_$name.json; }
run C1_len2560 --max-model-len 2560
run C2_sleep --sleep-mode
run C3_both_nowarm --max-model-len 2560 --sleep-mode --warm 0
echo "### qwen diag C DONE $(date -u +%H:%M)"
