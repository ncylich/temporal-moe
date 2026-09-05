#!/usr/bin/env bash
# Substitution tolerance (Appendix B): every matched temporal / full-MoE pair, eval only.
# One SUBSTEVAL process per checkpoint (analysis/probes/substitution_eval.py), all arms scored
# in-process on the same cached test micro-batches. Writes results/ablations/substitution/<run>.npz;
# analysis/residency/substitution_tolerance.py turns those into the CSV and the figure.
#
#   bash scripts/residency/orchestration/tmoe_substitution.sh            # all cells, skips done ones
#   ONLY=flame38m_g1_moe bash .../tmoe_substitution.sh                    # one cell
set -uo pipefail
cd "$(dirname "$0")/../../.."; . scripts/env.sh
export CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export EXTRA_ARGS="--no-rope-fusion --moe-use-legacy-grouped-gemm --dist-ckpt-strictness log_all"
export MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1
export SUBST_NSEQ=${SUBST_NSEQ:-512}
OUTD=$(pwd)/results/ablations/substitution; mkdir -p $OUTD   # absolute: the evaluator runs inside Megatron-LM/

# run  regime  SHAPE  FLOPS  GRAIN  MB  GB  corpus
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
moe_coarse_1e19          full      s19opt 1e19 1 16 1024 pythia
g1_tmoe_coarse_1e19      temporal  s19opt 1e19 1 16 1024 pythia
temporal_fine_g3_1e19    temporal  s19opt 1e19 3 16 1024 pythia
moe_fine_g3_1e19         full      s19opt 1e19 3 16 1024 pythia
g1_moe_s2_1e17           full      s2     1e17 1 32 256  tok16k
g1_tmoe_s2_1e17          temporal  s2     1e17 1 32 256  tok16k
"
echo "$CELLS" | grep -v '^\s*$' | while read -r run regime shape flops grain mb gb corpus; do
  [ -n "${ONLY:-}" ] && [ "$run" != "$ONLY" ] && continue
  out=$OUTD/$run.npz
  [ -f "$out" ] && { echo "[skip] $run done"; continue; }
  it=$(cat results/phase0/runs/$run/ckpt/latest_checkpointed_iteration.txt 2>/dev/null)
  [ -z "$it" ] && { echo "[skip] $run: no checkpoint pulled"; continue; }
  E=$((64 * grain)); k=$((6 * grain))
  if [ "$regime" = full ]; then R=$E; else R=$k; fi
  if [ "$corpus" = pythia ]; then
    tok=EleutherAI/pythia-12b; ddir=/root/data/dclm_tokenized
  else
    tok=$ROOT/data/tok16k; ddir=$ROOT/data/tok16k_full
  fi
  # Megatron sizes the test split as eval-iters x global batch; cover SUBST_NSEQ sequences.
  ei=$(( (SUBST_NSEQ + gb - 1) / gb ))
  echo "### subst $run regime=$regime E=$E k=$k R=$R ckpt=$it eval_iters=$ei $(date -u +%H:%M)"
  cp results/phase0/runs/$run/run.meta results/phase0/runs/$run/run.meta.presubst 2>/dev/null
  env SUBSTEVAL=1 RUN_NAME=$run SHAPE=$shape TARGET_FLOPS=$flops GRAIN=$grain MICRO_BATCH=$mb \
      GLOBAL_BATCH=$gb SEED=1234 TOKENIZER_MODEL=$tok DATA_DIR=$ddir \
      TEMPORAL_RESIDENCY_R=$R SUBST_REGIME=$regime SUBST_OUT=$out SUBST_EVAL_ITERS=$ei \
      timeout -k 60 ${SUBST_TIMEOUT:-3600} scripts/residency/gpu_lease.sh bash experiments/run.sh \
      > /workspace/rerun-logs/subst_$run.out 2>&1
  rc=$?
  mv -f results/phase0/runs/$run/run.meta.presubst results/phase0/runs/$run/run.meta 2>/dev/null
  grep -E "^\[subst\]" /workspace/rerun-logs/subst_$run.out | tail -4
  echo "### subst $run rc=$rc $(date -u +%H:%M)"
done
echo "### subst ALL DONE $(date -u +%H:%M)"
