#!/bin/bash
# Merge the faithful D7 published-recipe reconstruction, then score it on the FULL
# GSM8K test split. This is the arm the paper's adaptation claim rests on; at n=200
# it read -6.0 (identical to the unadapted base) with SE 3.0, which is uninformative
# either way. n=1319 decides whether the reconstruction heals R8 residency or not.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=$(NEED_GB=100 TIMEOUT=21600 scripts/residency/wait_for_gpu.sh) || exit 1
export CUDA_VISIBLE_DEVICES=$G
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-rebuild-merged
A=/workspace/olmoe-adapt/data/gemma_ce_rebuild_adapter.pt
scripts/residency/disk_budget.sh || exit 3
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 \
  --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
/workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
export HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_rebuild_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94
echo "### fullgsm rebuild DONE $(date -u +%H:%M)"
