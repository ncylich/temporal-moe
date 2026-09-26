#!/usr/bin/env bash
# Committed producer for results/ablations/unmask_eval_1e19.csv (README listed it as the one result
# without one). Each 1e19 checkpoint is scored in-process on the full 20-iteration test split
# (the canonical end-of-training test eval) in its native regime and in the crossed one:
# temporal -> every expert resident (unmask), full MoE -> rolling residency at R = k (impose).
# Also runs the two new cells: the fine full MoE and its imposition. SWEEP_SELFTEST re-scores the
# first arm at the end, which catches a consumed rather than replayed iterator.
set -uo pipefail
cd "$(dirname "$0")/../../.."; . scripts/env.sh
export CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export EXTRA_ARGS="--no-rope-fusion --moe-use-legacy-grouped-gemm --dist-ckpt-strictness log_all"
export MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1 SWEEP_SELFTEST=1
export TOKENIZER_MODEL=EleutherAI/pythia-12b DATA_DIR=/root/data/dclm_tokenized
# run  regime  GRAIN  R_native  R_cross
CELLS="
moe_coarse_1e19          full      1 64  6
g1_tmoe_coarse_1e19      temporal  1 6   64
temporal_fine_g3_1e19    temporal  3 18  192
moe_fine_g3_1e19         full      3 192 18
"
echo "$CELLS" | grep -v '^\s*$' | while read -r run regime grain rn rc; do
  [ -n "${ONLY:-}" ] && [ "$run" != "$ONLY" ] && continue
  if grep -q "^$run,native," results/ablations/sweep_eval.csv 2>/dev/null && grep -q "^$run,cross," results/ablations/sweep_eval.csv 2>/dev/null; then echo "[skip] $run done"; continue; fi
  echo "### unmask $run regime=$regime R_native=$rn R_cross=$rc $(date -u +%H:%M)"
  cp results/phase0/runs/$run/run.meta results/phase0/runs/$run/run.meta.preunmask 2>/dev/null
  env SWEEPEVAL=1 RUN_NAME=$run SHAPE=s19opt TARGET_FLOPS=1e19 GRAIN=$grain MICRO_BATCH=16 GLOBAL_BATCH=1024 SEED=1234 \
      TEMPORAL_RESIDENCY_R=$rn SWEEP="native:$rn cross:$rc" \
      timeout -k 60 ${UNMASK_TIMEOUT:-5400} scripts/residency/gpu_lease.sh bash experiments/run.sh > /workspace/rerun-logs/unmask_$run.out 2>&1
  rc_=$?; mv -f results/phase0/runs/$run/run.meta.preunmask results/phase0/runs/$run/run.meta 2>/dev/null
  grep -E "^\[sweep\]" /workspace/rerun-logs/unmask_$run.out | tail -5
  echo "### unmask $run rc=$rc_ $(date -u +%H:%M)"
done
echo "### unmask ALL DONE $(date -u +%H:%M)"
