#!/usr/bin/env bash
# Imposition dose curve at 1e19: rolling residency imposed on the two trained full MoEs at
# R = k, 2k, 4k, 8k and E (E = unconstrained), scored in-process on the full 20-iteration test
# split with the replay self-test. Rows land in results/ablations/sweep_eval.csv with tags
# imposeR<R>; analysis/residency/impose_dose_csv.py turns them into impose_dose_1e19.csv and the
# figure. Answers: how much resident memory does a full MoE need to reach the temporal model's
# quality at R = k?
set -uo pipefail
cd "$(dirname "$0")/../../.."; . scripts/env.sh
export CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export EXTRA_ARGS="--no-rope-fusion --moe-use-legacy-grouped-gemm --dist-ckpt-strictness log_all"
export MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1 SWEEP_SELFTEST=1
export TOKENIZER_MODEL=EleutherAI/pythia-12b DATA_DIR=/root/data/dclm_tokenized
CELLS="
moe_coarse_1e19   1 6  12 24 48 64
moe_fine_g3_1e19  3 18 36 72 144 192
"
echo "$CELLS" | grep -v '^\s*$' | while read -r run grain r1 r2 r3 r4 r5; do
  [ -n "${ONLY:-}" ] && [ "$run" != "$ONLY" ] && continue
  if grep -q "^$run,imposeR$r1," results/ablations/sweep_eval.csv 2>/dev/null && grep -q "^$run,imposeR$r4," results/ablations/sweep_eval.csv 2>/dev/null; then echo "[skip] $run done"; continue; fi
  echo "### impose $run R=$r1,$r2,$r3,$r4,$r5 $(date -u +%H:%M)"
  cp results/phase0/runs/$run/run.meta results/phase0/runs/$run/run.meta.preimpose
  env SWEEPEVAL=1 RUN_NAME=$run SHAPE=s19opt TARGET_FLOPS=1e19 GRAIN=$grain MICRO_BATCH=16 GLOBAL_BATCH=1024 SEED=1234 \
      TEMPORAL_RESIDENCY_R=$r5 SWEEP="imposeR$r1:$r1 imposeR$r2:$r2 imposeR$r3:$r3 imposeR$r4:$r4 imposeR$r5:$r5" \
      timeout -k 60 ${IMPOSE_TIMEOUT:-7200} scripts/residency/gpu_lease.sh bash experiments/run.sh > /workspace/rerun-logs/impose_$run.out 2>&1
  rc=$?; mv -f results/phase0/runs/$run/run.meta.preimpose results/phase0/runs/$run/run.meta
  grep -E "^\[sweep\] (impose|SELFTEST)" /workspace/rerun-logs/impose_$run.out | cut -c1-110
  echo "### impose $run rc=$rc $(date -u +%H:%M)"
done
echo "### impose ALL DONE $(date -u +%H:%M)"
