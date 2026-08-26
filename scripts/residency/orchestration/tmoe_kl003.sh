#!/bin/bash
# KL 0.03 -- the untested lever gemma_adapt_RESULTS section Open names.
#
# The ladder makes KL weight the free-arm/constrained-arm dial:
#   no KL  (D7)  -> R8 MMLU -0.7 but a weak free arm
#   KL 0.1 (D8)  -> free arm repaired, R8 MMLU -2.9
#   KL 0.05 (D12)-> interpolates; strongest constrained row of the program
# Our failure is the CONSTRAINED arm on GSM8K, so the direction to try is LESS anchoring
# to base, giving the adapter more room to move toward the constraint. 0.03 is the lower
# end of the bracket the RESULTS file proposes and never ran.
#
# Single variable: identical pool, identical trajectories, identical 3.4M budget, r32,
# micro-batch 16. Only --kl-weight differs from arm C.
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2
echo "### kl003 TRAIN $(date -u +%H:%M)"
/workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py \
  --model /dev/shm/gemma4-26b-it --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 \
  --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --accum 16 --lr 3e-5 \
  --tokens 3400000 --kl-anchor /workspace/instruct-traj/gemma4_d7_seq4096_klref.pt \
  --kl-weight 0.03 --out /workspace/olmoe-adapt/data/gemma_ce_kl003_adapter.pt
echo "### kl003 TRAIN DONE $(date -u +%H:%M)"
/workspace/venv_fla/bin/python scripts/residency/mirror_artifact.py \
  --path /workspace/olmoe-adapt/data/gemma_ce_kl003_adapter.pt --kind adapter
echo "### kl003 ALL DONE $(date -u +%H:%M)"
