#!/usr/bin/env bash
# qwen parity diagnosis: (B) is the merged checkpoint's eager greedy output reproducible across engine memory
# budgets (0.85 reference vs the sampler's 0.55)? (A) after the sampler's sync, are the engine tensors bit-exact
# with the merged checkpoint (check_engine), and how many greedy generations match?
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TEMPORAL_EAGER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
S=/tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad; L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
echo "### qwen diag B: merged checkpoint standalone at gpu-mem 0.55 vs the 0.85 reference $(date -u +%H:%M)"
$L $PY -u analysis/residency/parity_vllm.py --path /root/models/qwen35-digit3-merged --R 8 --n 8 --max-new 256 --think off --gpu-mem 0.55 --out $S/ref_qwen_digit3_eager_mem055.json
$PY -u analysis/residency/parity_vllm.py --compare $S/ref_qwen_digit3_eager.json $S/ref_qwen_digit3_eager_mem055.json
echo "### qwen diag A: sampler sync -> exact tensor check + greedy parity $(date -u +%H:%M)"
cp /workspace/olmoe-adapt/data/qwen_ce_digit3_adapter.pt /tmp/smoke_qwen_adapter.pt
$L $PY -u analysis/residency/train_gemma_ce.py --model /root/models/qwen35-35b-a3b --family qwen35 --no-unsloth --traj qwen35_d7_seq4096 --max-seq 4096 \
  --expert-lora-r 16 --opt adamw --micro-batch 16 --out /tmp/smoke_qwen_adapter.pt --resume --tokens 1 \
  --online-every 16 --online-n 32 --online-gpu-mem 0.55 --online-offload 20 --online-smoke $S/ref_qwen_digit3_eager.json
echo "### qwen diag DONE rc=$? $(date -u +%H:%M)"
