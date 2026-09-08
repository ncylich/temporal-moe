#!/usr/bin/env bash
# Expert output similarity (mechanism check behind substitution tolerance) over every checkpoint
# of the substitution matrix, plus one random-init cell per shape and grain for calibration
# (RUN_NAME without a checkpoint: Megatron starts from init). analysis/probes/expert_similarity.py
# writes results/ablations/expert_similarity/<run>.npz; analysis/residency/expert_similarity.py
# aggregates them.
set -uo pipefail
cd "$(dirname "$0")/../../.."; . scripts/env.sh
export CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export EXTRA_ARGS="--no-rope-fusion --moe-use-legacy-grouped-gemm --dist-ckpt-strictness log_all"
export MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1 EXPSIM_N=${EXPSIM_N:-2048}
OUTD=$(pwd)/results/ablations/expert_similarity; mkdir -p $OUTD
# run  regime  SHAPE  FLOPS  GRAIN  MB  GB  corpus   (regime "init" = random initialisation, no checkpoint)
CELLS="
flame38m_g1_moe          full      s38m   1e18 1 32 1024 pythia
flame38m_g1_temporal     temporal  s38m   1e18 1 32 1024 pythia
flame38m_g3_moe          full      s38m   1e18 3 32 1024 pythia
flame38m_g3_temporal     temporal  s38m   1e18 3 32 1024 pythia
flame38m_g1_moe_s2       full      s38m   1e18 1 32 1024 pythia
flame38m_g1_temporal_s2  temporal  s38m   1e18 1 32 1024 pythia
flame38m_g3_moe_s2       full      s38m   1e18 3 32 1024 pythia
flame38m_g3_temporal_s2  temporal  s38m   1e18 3 32 1024 pythia
flame38m_g1_moe_s3       full      s38m   1e18 1 32 1024 pythia
flame38m_g1_temporal_s3  temporal  s38m   1e18 1 32 1024 pythia
flame38m_g3_moe_s3       full      s38m   1e18 3 32 1024 pythia
flame38m_g3_temporal_s3  temporal  s38m   1e18 3 32 1024 pythia
g3_moe_s2_1e17           full      s2     1e17 3 32 256  tok16k
g3_tmoe_s2_1e17          temporal  s2     1e17 3 32 256  tok16k
g1_moe_s2_1e17           full      s2     1e17 1 32 256  tok16k
g1_tmoe_s2_1e17          temporal  s2     1e17 1 32 256  tok16k
moe_coarse_1e19          full      s19opt 1e19 1 16 1024 pythia
g1_tmoe_coarse_1e19      temporal  s19opt 1e19 1 16 1024 pythia
temporal_fine_g3_1e19    temporal  s19opt 1e19 3 16 1024 pythia
moe_fine_g3_1e19         full      s19opt 1e19 3 16 1024 pythia
init_s2_g1               init      s2     1e17 1 32 256  tok16k
init_s2_g3               init      s2     1e17 3 32 256  tok16k
init_s38m_g1             init      s38m   1e18 1 32 1024 pythia
init_s38m_g3             init      s38m   1e18 3 32 1024 pythia
init_s19opt_g1           init      s19opt 1e19 1 16 1024 pythia
init_s19opt_g3           init      s19opt 1e19 3 16 1024 pythia
"
echo "$CELLS" | grep -v '^\s*$' | while read -r run regime shape flops grain mb gb corpus; do
  [ -n "${ONLY:-}" ] && [ "$run" != "$ONLY" ] && continue
  out=$OUTD/$run.npz; [ -f "$out" ] && { echo "[skip] $run done"; continue; }
  if [ "$regime" != init ] && [ ! -f results/phase0/runs/$run/ckpt/latest_checkpointed_iteration.txt ]; then echo "[skip] $run: no checkpoint"; continue; fi
  E=$((64 * grain)); k=$((6 * grain)); if [ "$regime" = temporal ]; then R=$k; else R=$E; fi
  if [ "$corpus" = pythia ]; then tok=EleutherAI/pythia-12b; ddir=/root/data/dclm_tokenized; else tok=$ROOT/data/tok16k; ddir=$ROOT/data/tok16k_full; fi
  ei=1
  echo "### expsim $run regime=$regime E=$E k=$k R=$R $(date -u +%H:%M)"
  [ -f results/phase0/runs/$run/run.meta ] && cp results/phase0/runs/$run/run.meta results/phase0/runs/$run/run.meta.preexpsim
  env EXPERTSIM=1 RUN_NAME=$run SHAPE=$shape TARGET_FLOPS=$flops GRAIN=$grain MICRO_BATCH=$mb GLOBAL_BATCH=$gb SEED=1234 \
      TOKENIZER_MODEL=$tok DATA_DIR=$ddir TEMPORAL_RESIDENCY_R=$R EXPSIM_REGIME=$regime EXPSIM_OUT=$out SUBST_EVAL_ITERS=$ei \
      timeout -k 60 ${EXPSIM_TIMEOUT:-1800} scripts/residency/gpu_lease.sh bash experiments/run.sh > /workspace/rerun-logs/expsim_$run.out 2>&1
  rc=$?; [ -f results/phase0/runs/$run/run.meta.preexpsim ] && mv -f results/phase0/runs/$run/run.meta.preexpsim results/phase0/runs/$run/run.meta
  [ "$regime" = init ] && rm -rf results/phase0/runs/$run
  grep -E "^\[expsim\] (L 2|wrote)" /workspace/rerun-logs/expsim_$run.out | cut -c1-120
  echo "### expsim $run rc=$rc $(date -u +%H:%M)"
done
echo "### expsim ALL DONE $(date -u +%H:%M)"
