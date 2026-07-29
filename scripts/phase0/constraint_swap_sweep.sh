#!/bin/bash
# X1 / X2 -- cross-regime constraint swap, global and per layer. Evaluation only, no training.
#
# X1 (global): evaluate a trained model under the other regime. Unmasking a temporal model means
# making every expert always resident (R=E); the converse imposes rolling residency on an
# unconstrained checkpoint (R=k). This reproduces section 5 of delexicalization.md, which was run ad
# hoc through run.sh EVAL_ONLY=1 with the temporal env flags and never had a committed driver.
#
# X2 (per layer) is test C3 of LAYER_LEXICALITY.md, and the reason this driver exists: swap the
# constraint at ONE layer at a time and sweep the layer. It yields a per-layer cost profile for 2L
# evaluation passes and no training, and it is deterministic -- same checkpoint, same fixed batch, no
# seed noise -- so even small differences are readable. Its shape is the gate on whether the T1-T3
# training sweeps are worth their budget.
#
#   scripts/phase0/constraint_swap_sweep.sh --global RUN            # X1: native vs other regime
#   scripts/phase0/constraint_swap_sweep.sh --per-layer RUN         # X2/C3: one layer at a time
#   LAYERS="2 3 4" scripts/phase0/constraint_swap_sweep.sh --per-layer RUN
#
# What the direction means, per trained regime:
#   temporal checkpoint  -> UNMASK layer l (R=E there, R=k elsewhere). Cost of giving back freedom.
#   full-MoE checkpoint  -> IMPOSE residency at layer l (R=k there, R=E elsewhere). Cost of taking
#                           freedom away from a model trained expecting it.
# Both run through temporal/pretrain_temporal.py, which installs the residency router; the per-layer
# schedule is TEMPORAL_R_SCHEDULE (see temporal/temporal_router.py).
#
# The limitation is the one section 5 of delexicalization.md already identified and it is not fixable
# here: there is no co-adaptation, so this measures the cost of *removing* freedom from a model that
# was trained expecting it, not the cost of never having had it. That makes it an upper bound whose
# shape is still informative.
#
# Weights are frozen (--lr 0 --min-lr 0 via EVAL_ONLY's temporal path). Results are appended to
# results/ablations/swap_sweep.csv; the CE is parsed from the eval log.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$ROOT"

MODE=${1:?"usage: $0 --global|--per-layer RUN [RUN ...]"}
shift
[ $# -ge 1 ] || { echo "usage: $0 $MODE RUN [RUN ...]" >&2; exit 1; }

RUNS_DIR=${CKPT_ROOT:-$ROOT/results/phase0/runs}
OUTCSV=$ROOT/results/ablations/swap_sweep.csv
[ -f "$OUTCSV" ] || echo "run,budget,regime,grain,arm,layer,R_at_layer,R_elsewhere,test_CE,test_BPB,log" > "$OUTCSV"

# BPB divisor: bytes-per-token of the eval corpus, as used by every published BPB in this repo.
BPB_DIV=${BPB_DIV:-2.9780}

# One field out of a run.meta line, either format.
meta_get() { sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" "$1" | head -1; }

run_one() {   # run, arm, layer(or "-"), R_at_layer, R_elsewhere, schedule, logname
  local run=$1 arm=$2 layer=$3 rat=$4 relse=$5 sched=$6 logname=$7
  local meta="$RUNS_DIR/$run/run.meta"
  local shape grain topk mb gb flops smult
  shape=${SHAPE:-$(meta_get "$meta" shape)}
  grain=$(meta_get "$meta" grain); topk=$(meta_get "$meta" topk)
  mb=$(meta_get "$meta" mb); gb=$(meta_get "$meta" gb)
  flops=${TARGET_FLOPS:-$(meta_get "$meta" flops)}; smult=$(meta_get "$meta" shared_mult)
  if [ -z "$shape" ] || [ -z "$flops" ]; then
    echo "[skip] $run: run.meta has no shape=/flops=; pass SHAPE= and TARGET_FLOPS=" >&2
    return 0
  fi
  cp "$meta" "$meta.presweep"
  echo "--- $run [$arm] layer=$layer R@layer=$rat R_elsewhere=$relse  schedule='$sched'"
  set +e
  env EVAL_ONLY=1 TEMPORAL=1 RUN_NAME="$run" SHAPE="$shape" TARGET_FLOPS="$flops" \
      GRAIN="${grain:-1}" TOPK="$topk" MICRO_BATCH="$mb" GLOBAL_BATCH="$gb" \
      SHARED_MULT="${smult:-2}" \
      TEMPORAL_RESIDENCY_R="$relse" TEMPORAL_R_SCHEDULE="$sched" \
      TEMPORAL_EVICT="${TEMPORAL_EVICT:-min_logit}" EVAL_ITERS="${EVAL_ITERS:-16}" \
      ./experiments/run.sh > "$RUNS_DIR/$run/$logname" 2>&1
  local rc=$?
  set -e
  # run.sh rewrites run.meta before the mode branch; restore the published provenance.
  cmp -s "$meta" "$meta.presweep" || echo "[warn] $run: run.meta was rewritten, restoring"
  mv -f "$meta.presweep" "$meta"
  if [ $rc -ne 0 ]; then
    echo "[fail] $run [$arm]: exit $rc, see $RUNS_DIR/$run/$logname" >&2
    return 0
  fi
  # Megatron prints the test-set loss as "validation loss at ... | lm loss value: X" on the test pass.
  local ce
  ce=$(grep -oE "lm loss value: [0-9.]+" "$RUNS_DIR/$run/$logname" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "$ce" ]; then
    echo "[warn] $run [$arm]: no lm loss in log; recorded blank" >&2
  fi
  local bpb=""
  [ -n "$ce" ] && bpb=$("$PY" -c "print(f'{$ce/0.6931471805599453/$BPB_DIV:.4f}')" 2>/dev/null || true)
  local budget regime grain_l
  read -r budget regime grain_l < <("$PY" - "$run" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.environ["ROOT"], "analysis", "probes"))
import registry
r = registry.get(sys.argv[1])
print(r.budget, r.regime, r.grain_label.replace(" ", ""))
PY
)
  echo "$run,$budget,$regime,$grain_l,$arm,$layer,$rat,$relse,${ce:-},${bpb:-},$logname" >> "$OUTCSV"
  echo "[ok]   $run [$arm] layer=$layer  test CE=${ce:-?}  BPB=${bpb:-?}"
}

for run in "$@"; do
  meta="$RUNS_DIR/$run/run.meta"
  [ -f "$meta" ] || { echo "[skip] $run: no run.meta"; continue; }
  [ -d "$RUNS_DIR/$run/ckpt" ] || { echo "[skip] $run: no checkpoint on disk"; continue; }
  k=$(meta_get "$meta" topk)
  E=$(meta_get "$meta" num_experts)
  temporal=$(meta_get "$meta" temporal)
  mode=$(meta_get "$meta" mode)
  [ -z "$temporal" ] && { [ "$mode" = "temporal" ] && temporal=1 || temporal=0; }
  depth=$("$PY" -c "
import sys, os
sys.path.insert(0, os.path.join(os.environ['ROOT'], 'analysis', 'probes'))
import registry
print(registry.depth_of('$run') or 0)")
  if [ "$depth" -lt 2 ]; then
    echo "[skip] $run: transformer depth unknown, cannot enumerate MoE layers"; continue
  fi
  # MoE layers are 2..depth: layer 1 is a dense FFN in every config.
  layers=${LAYERS:-$(seq 2 "$depth")}

  if [ "$MODE" = "--global" ]; then
    if [ "$temporal" = "1" ]; then
      run_one "$run" native      - "$k" "$k" "" "swap_native.log"
      run_one "$run" unmask_all  - "$E" "$E" "" "swap_unmask_all.log"
    else
      run_one "$run" native      - "$E" "$E" "" "swap_native.log"
      run_one "$run" impose_all  - "$k" "$k" "" "swap_impose_all.log"
    fi
  elif [ "$MODE" = "--per-layer" ]; then
    # Native reference first, so every per-layer delta is against a number measured in this same
    # environment rather than against a published one.
    if [ "$temporal" = "1" ]; then
      run_one "$run" native - "$k" "$k" "" "swap_native.log"
      for l in $layers; do
        run_one "$run" unmask_one "$l" "$E" "$k" "$l:E" "swap_unmask_L$l.log"
      done
    else
      run_one "$run" native - "$E" "$E" "" "swap_native.log"
      for l in $layers; do
        run_one "$run" impose_one "$l" "$k" "$E" "$l:$k" "swap_impose_L$l.log"
      done
    fi
  else
    echo "unknown mode $MODE (expected --global or --per-layer)" >&2; exit 1
  fi
done

echo
echo "wrote $OUTCSV"
"$PY" - <<'PY'
import csv, os
p = os.path.join(os.environ["ROOT"], "results/ablations/swap_sweep.csv")
rows = list(csv.DictReader(open(p)))
print(f"{len(rows)} rows; per-layer arms:")
for r in rows:
    if r["arm"] in ("unmask_one", "impose_one"):
        print(f"  {r['run']:26} L{r['layer']:<3} {r['arm']:11} CE={r['test_CE'] or '?':9} "
              f"BPB={r['test_BPB'] or '?'}")
PY
