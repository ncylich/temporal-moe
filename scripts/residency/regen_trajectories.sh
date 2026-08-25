#!/bin/bash
# RECOVER_DATA_PLAN section 1.2 -- regenerate the self-generated CE trajectories.
#
#     regen_trajectories.sh gemma    # GPU 1
#     regen_trajectories.sh qwen     # GPU 2
#
# Each model generates its OWN think-off responses to the rebuilt d7 pool, under NO
# residency constraint, sampled per its own generation config with seed 1234. That is the
# recipe: the adapter then trains on these with plain CE *with* the constraint on, so the
# trajectories must be what the free model would really have said.
#
# Cap is 8192, deliberately far above the recipe's 2048/3072. The cut is decided AFTER
# generation from the measured length distribution on this pool rather than inherited from
# a benchmark-derived guess -- vLLM stops at EOS, so a high cap costs almost nothing except
# on genuinely long rows. Rows are dropped WHOLE at the chosen threshold, never truncated:
# gemma_adapt_RESULTS records that mid-response cuts teach degenerate early endings
# (7 to 13-token IFEval answers).
#
# --max-prompt-tok MUST match the pool builder's gate (1024) or rows are silently dropped
# here, after the pool has already been counted and committed.
set -euo pipefail

ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd $ROOT

PY=/workspace/venv_vllm312/bin/python
GEN=analysis/residency/gen_traj_vllm.py
POOL=${POOL:-/workspace/olmoe-adapt/data/d7_prompts.jsonl}
OUT=${OUT:-/workspace/instruct-traj}
LOG=${LOG_DIR:-/workspace/rerun-logs}
MAXNEW=${MAXNEW:-8192}
PROMPT_TOK=${PROMPT_TOK:-1024}
mkdir -p $OUT $LOG

[ -s "$POOL" ] || { echo "pool missing: $POOL" >&2; exit 2; }

# think-on lanes are the B2 prerequisite (RECOVER_DATA_PLAN Part 0 group B): Section 7's
# length result rests on gemma alone because qwen's adapted checkpoints were only ever
# evaluated think-off, where neither routing regime lengthens and the comparison is empty.
# They need a bigger envelope than think-off -- gemma_adapt_RESULTS records 35.7% of
# think-on responses exceeding 3072 with a median think response of 2346 -- which the
# 8192 cap already covers. GPU is overridable; this box is shared.
case "${1:?usage: regen_trajectories.sh gemma|qwen|qwen-think|gemma-think}" in
  gemma) DEV=${GPU:-1}; MODEL=/dev/shm/gemma4-26b-it;  TAG=gemma4_d7;  EXTRA=() ;;
  qwen)  DEV=${GPU:-2}; MODEL=/dev/shm/qwen35-35b-a3b; TAG=qwen35_d7;  EXTRA=(--max-seqs 128) ;;
  qwen-think)  DEV=${GPU:-1}; MODEL=/dev/shm/qwen35-35b-a3b; TAG=qwen35_d7think
               EXTRA=(--max-seqs 128 --think on) ;;
  gemma-think) DEV=${GPU:-1}; MODEL=/dev/shm/gemma4-26b-it;  TAG=gemma4_d7think
               EXTRA=(--think on) ;;
  *) echo "unknown: $1" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=$DEV
echo "### traj $TAG on GPU $DEV, cap $MAXNEW $(date -u +%H:%M)"
case "$1" in *-think) THINK=() ;; *) THINK=(--think off) ;; esac
$PY -u $GEN --model $MODEL --tag $TAG --prompts $POOL \
    "${THINK[@]}" --max-new $MAXNEW --max-prompt-tok $PROMPT_TOK \
    --gpu-mem 0.94 --out $OUT "${EXTRA[@]}" 2>&1 | tee $LOG/traj_${TAG}.log
echo "### traj $TAG DONE $(date -u +%H:%M)"

# Durability BEFORE any training reads it. The whole recovery plan exists because this
# step was skipped in August: every artifact lived only on a pod disk, and the four HF
# repos were last written three weeks before the program ran.
echo "### mirroring $TAG to Hugging Face $(date -u +%H:%M)"
$PY $ROOT/scripts/residency/mirror_artifact.py --path $OUT/${TAG}.pt --kind trajectory
