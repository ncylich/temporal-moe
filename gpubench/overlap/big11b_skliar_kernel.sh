#!/usr/bin/env bash
# 11 B random-weight model (uniform Q4_0 file), fused op (TEMPORAL_FUSED=1), streamed: ours against Skliar with
# Skliar's own routing kernel (k_skliar) in the timed path. big11b.sh ran both arms on our routing kernel.
# Loads are still imposed (TEMPORAL_SWAP_HIST), because a random router decides nothing: k_skliar runs on the
# 192 logits of every layer-token and does its own work, then the imposed swap list replaces the one it wrote.
#   one        our kernel, exactly one load per layer-token
#   sk         k_skliar at the 1.26 B model's lambda, loads drawn from Skliar's histogram on the 1.26 B model
#   skmin      the same with a lambda so large that no expert ever wins on merit: k_skliar's least work
#              (one round, forced top-1 only)
#   skshape    our kernel with Skliar's histogram (the arm of big11b.sh, to tie this session to that one)
#   nb-*       the same kernels with no bytes moved (TEMPORAL_ROUTE_NOCOPY=1); these print each kernel's own
#              load histogram on the random logits, which says how many rounds k_skliar ran
#   stub       routing kernel replaced by an empty launch; skipAB: stub-free op removed (TEMPORAL_FUSED_SKIP=3)
# Protocol of PRETRAINED_LEDGER P1-12. Passes alternate the arm order.
#   big11b_skliar_kernel.sh <llama.cpp-dir> <model.gguf> <out-dir> <skliar histogram> <lambda> [passes]
set -uo pipefail
BIN="$1/build/bin/llama-bench"; M="$2"; OUT="$3"; SK="$4"; LAM="$5"; PASSES="${6:-3}"
mkdir -p "$OUT"
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
run() {   # tag, reps, env...
  local tag="$1" r="$2"; shift 2
  env TEMPORAL_UNIFIED=1 TEMPORAL_FUSED=1 "$@" taskset -c $CPUS "$BIN" -m "$M" -ngl 99 -fa 1 -ncmoe 48 -ub 1 -b 1 -d 1024 -n 128 -r $r -o json > "$OUT/$tag.json" 2> "$OUT/$tag.err"
  python3 - "$OUT" "$tag" <<'PY'
import json, sys, re
o, t = sys.argv[1:]
try:
    r = json.load(open(f"{o}/{t}.json"))[-1]
    e = open(f"{o}/{t}.err").read()
    h = re.findall(r"temporal-swap-hist: (.*)", e)
    n = re.findall(r"loads_per_layer_token=([0-9.]+).*?hist=\[([0-9,]*)\]", e)
    print(f"{t}: {r['avg_ts']:.2f} tok/s (sd {r['stddev_ts']:.2f}) {h[-1] if h else ''} {('own ' + n[-1][0] + ' [' + n[-1][1] + ']') if n else ''}", flush=True)
except Exception as ex:
    print(f"{t}: FAILED {ex}", flush=True)
PY
}
D="LLAMA_TEMPORAL_ROUTE=decode"; S="LLAMA_TEMPORAL_ROUTE=skliar TEMPORAL_CB_C=18 TEMPORAL_CB_J=1"
L=("one|$D TEMPORAL_SWAP_HIST=0,1" "sk|$S TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_SWAP_HIST=$SK"
   "skmin|$S TEMPORAL_CB_LAMBDA=1000000 TEMPORAL_SWAP_HIST=$SK" "skshape|$D TEMPORAL_SWAP_HIST=$SK"
   "nb-ours|$D TEMPORAL_ROUTE_NOCOPY=1" "nb-sk|$S TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_ROUTE_NOCOPY=1"
   "nb-skmin|$S TEMPORAL_CB_LAMBDA=1000000 TEMPORAL_ROUTE_NOCOPY=1" "stub|$D TEMPORAL_ROUTE_STUB=1"
   "skipAB|$D TEMPORAL_ROUTE_NOCOPY=1 TEMPORAL_FUSED_SKIP=3")
for p in $(seq 1 "$PASSES"); do
  ORDER=$(seq 0 $((${#L[@]} - 1))); [ $((p % 2)) = 0 ] && ORDER=$(seq $((${#L[@]} - 1)) -1 0)
  for i in $ORDER; do
    IFS='|' read -r tag E <<< "${L[$i]}"
    run "p$p-$tag" 8 $E
  done
  echo "### pass $p done $(date -u +%T)"
  if [ "$p" = 1 ]; then   # k_skliar's own load histogram on the random logits at other lambdas (2 repetitions, not timed rows)
    for l in ${SWEEP:-0.3 0.5 1.0 1.5 2.5}; do run "sweep-lam$l" 2 $S TEMPORAL_CB_LAMBDA=$l TEMPORAL_ROUTE_NOCOPY=1; done
  fi
done
echo RUN_DONE
