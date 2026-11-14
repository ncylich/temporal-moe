#!/bin/bash
# C8 / N6 -- causal token-versus-context substitution. Runs the three arms for each named run and then
# the comparison. See analysis/probes/delex_causal.py for the design and why each arm is constructed
# the way it is.
#
#   scripts/phase0/causal_substitution.sh RUN [RUN ...]
#
# Three forward passes per run, each a couple of minutes. The arms are separate invocations of the same
# fixed batch; delex_causal.py --analyze records an input-id hash per arm and refuses to compare arms
# that did not see the same batch, so the assumption is checked rather than trusted.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$ROOT"

RUNS_DIR=${CKPT_ROOT:-$ROOT/results/phase0/runs}
[ $# -ge 1 ] || { echo "usage: $0 RUN [RUN ...]" >&2; exit 1; }

meta_get() { sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" "$1" | head -1; }

ok=0; failed=0
for run in "$@"; do
  meta="$RUNS_DIR/$run/run.meta"
  [ -f "$meta" ] || { echo "[skip] $run: no run.meta"; continue; }
  [ -d "$RUNS_DIR/$run/ckpt" ] || { echo "[skip] $run: no checkpoint"; continue; }
  shape=$(meta_get "$meta" shape); flops=$(meta_get "$meta" flops)
  [ -z "$shape" ] && shape=$("$PY" -c "
import sys, os; sys.path.insert(0, os.path.join('$ROOT','analysis','probes'))
import registry; print(registry.shape_of('$run') or '')")
  [ -z "$flops" ] && flops=$("$PY" -c "
import sys, os; sys.path.insert(0, os.path.join('$ROOT','analysis','probes'))
import registry; b=registry.get('$run').budget; print('' if b=='unknown' else b)")
  grain=$(meta_get "$meta" grain); topk=$(meta_get "$meta" topk)
  mb=$(meta_get "$meta" mb); gb=$(meta_get "$meta" gb); smult=$(meta_get "$meta" shared_mult)
  temporal=$(meta_get "$meta" temporal); mode=$(meta_get "$meta" mode)
  [ -z "$temporal" ] && { [ "$mode" = "temporal" ] && temporal=1 || temporal=0; }
  if [ -z "$shape" ] || [ -z "$flops" ]; then
    echo "[skip] $run: no shape/budget determinable"; continue
  fi

  for arm in ref token context; do
    cp "$meta" "$meta.presweep"
    echo "--- $run [$arm]"
    set +e
    env CAUSALPROBE=1 CAUSAL_ARM="$arm" RUN_NAME="$run" SHAPE="$shape" TARGET_FLOPS="$flops" \
        GRAIN="${grain:-1}" TOPK="$topk" MICRO_BATCH="$mb" GLOBAL_BATCH="$gb" \
        SHARED_MULT="${smult:-2}" TEMPORAL="$temporal" \
        TEMPORAL_EVICT="${TEMPORAL_EVICT:-min_logit}" \
        ./experiments/run.sh > "$RUNS_DIR/$run/causal_${arm}_run.log" 2>&1
    rc=$?
    set -e
    cmp -s "$meta" "$meta.presweep" || echo "[warn] $run: run.meta rewritten, restoring"
    mv -f "$meta.presweep" "$meta"
    if [ $rc -ne 0 ] || [ ! -f "$RUNS_DIR/$run/causal_${arm}.pt" ]; then
      echo "[fail] $run [$arm]: exit $rc, see $RUNS_DIR/$run/causal_${arm}_run.log"
      failed=$((failed+1)); continue
    fi
    echo "[ok]   $run [$arm]"
    ok=$((ok+1))
  done
done

echo
echo "=== causal arms: $ok ok, $failed failed ==="
"$PY" analysis/probes/delex_causal.py --analyze "$@"
