#!/usr/bin/env bash
# Do the residency hooks change the FREE arm? Plain vLLM (no hooks) vs the hooked engine, greedy, eager.
# qwen: raw base + W=3 adapter vs the existing hooked reference. gemma: base, hooked then plain.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TEMPORAL_EAGER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
S=/tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad; L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
echo "### hook diag 1/3 qwen raw+adapter, NO hooks, free arm $(date -u +%H:%M)"
$L $PY -u analysis/residency/parity_vllm.py --path /root/models/qwen35-35b-a3b --adapter /workspace/olmoe-adapt/data/qwen_ce_digit3_adapter.pt --R 8 --n 8 --max-new 256 --think off --gpu-mem 0.85 --no-hooks --out $S/ref_qwen_nohooks.json
$PY analysis/residency/parity_vllm.py --compare $S/ref_qwen_digit3_eager.json $S/ref_qwen_nohooks.json
echo "### hook diag 2/3 gemma base, hooks, free+R8 $(date -u +%H:%M)"
$L $PY -u analysis/residency/parity_vllm.py --path /dev/shm/gemma4-26b-it --R 8 --n 8 --max-new 256 --gpu-mem 0.85 --out $S/ref_gemma_hooks.json
echo "### hook diag 3/3 gemma base, NO hooks, free $(date -u +%H:%M)"
$L $PY -u analysis/residency/parity_vllm.py --path /dev/shm/gemma4-26b-it --R 8 --n 8 --max-new 256 --gpu-mem 0.85 --no-hooks --out $S/ref_gemma_nohooks.json
$PY analysis/residency/parity_vllm.py --compare $S/ref_gemma_hooks.json $S/ref_gemma_nohooks.json
echo "### hook diag DONE $(date -u +%H:%M)"
