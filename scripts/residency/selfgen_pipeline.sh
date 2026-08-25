#!/bin/bash
# The whole self-generated-lane experiment, end to end, unattended.
#
# Hypothesis: constraint-robustness on math and code comes from training on the model's OWN
# generated problems. Evidence so far, on two architectures: with real-corpus math and code
# lanes, IFEval / MMLU / (qwen) HumanEval reproduce or beat published, while GSM8K fails on
# both (-5.5 vs +0.0 gemma, -10.0 vs -3.5 qwen) and gemma HumanEval only partly recovers.
# The two regressed lanes are exactly the two the ORIGINAL pool generated itself.
#
# This rebuilds the pool with self-generated math and code substituted in, then repeats the
# full chain: trajectories -> train -> merge -> verify -> grid, for both models.
#
#     selfgen_pipeline.sh          # runs everything
set -u
ROOT=/workspace/temporal-moe
LOG=/workspace/rerun-logs
DATA=/workspace/olmoe-adapt/data
cd $ROOT
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
say () { echo "### PIPE $* $(date -u +%H:%M)"; }

say "1/6 rebuild pool with self-generated math + code"
/workspace/venv_fla/bin/python -u analysis/residency/build_d7_prompts.py \
    --out $DATA --scan-cap 1000000 \
    --selfgen-math $DATA/selfgen_math_prompts.jsonl \
    --selfgen-code $DATA/selfgen_code_prompts.jsonl > $LOG/pipe_pool.log 2>&1
grep -E "^\[d7\]|WARNING" $LOG/pipe_pool.log | tail -6
[ -s $DATA/d7_prompts.jsonl ] || { echo "### PIPE ABORT: pool not built"; exit 1; }

say "2/6 trajectories, both models in parallel"
GPU=2 POOL=$DATA/d7_prompts.jsonl OUT=/workspace/instruct-traj \
  $ROOT/scripts/residency/regen_trajectories.sh gemma > $LOG/pipe_traj_gemma.log 2>&1 &
G=$!
GPU=3 POOL=$DATA/d7_prompts.jsonl OUT=/workspace/instruct-traj \
  $ROOT/scripts/residency/regen_trajectories.sh qwen  > $LOG/pipe_traj_qwen.log 2>&1 &
Q=$!
wait $G $Q
say "2/6 trajectories done"

say "3/6 cut trajectories to the training sequence, whole rows only"
for t in gemma4_d7 qwen35_d7; do
  /workspace/venv_fla/bin/python analysis/residency/cut_trajectories.py --tag $t --max-seq 4096 \
    2>&1 | grep -E '"rows_kept"|"rows_dropped"|"truncated'
done

say "4/6 retrain both adapters"
# fresh KL anchors: the trajectory set changed, so the cached ones are stale
rm -f /workspace/instruct-traj/*_seq4096_klref.pt
GPU=2 $ROOT/scripts/residency/train_adapters.sh gemma > $LOG/pipe_train_gemma.log 2>&1 &
G=$!
GPU=3 $ROOT/scripts/residency/train_adapters.sh qwen  > $LOG/pipe_train_qwen.log 2>&1 &
Q=$!
wait $G $Q
say "4/6 training done"

say "5/6 merge, verify, grid"
GPU=2 $ROOT/scripts/residency/merge_and_remeasure.sh gemma > $LOG/pipe_grid_gemma.log 2>&1 &
G=$!
GPU=3 $ROOT/scripts/residency/remeasure_qwen.sh > $LOG/pipe_grid_qwen.log 2>&1 &
Q=$!
wait $G $Q

say "6/6 gemma channel-aware HumanEval"
GPU=2 $ROOT/scripts/residency/remeasure_humaneval.sh gemma > $LOG/pipe_he_gemma.log 2>&1
say "PIPELINE COMPLETE"
