#!/usr/bin/env bash
# 11 B random-weight Qwen3-MoE (fine: 18 of 192 experts, 45 layers, Q4_K_M, no shared expert) on one GPU:
# what a load costs with engine patch 0013's switch off and on, at forced load counts. The router of a
# random-weight model decides nothing, so the loads are forced on the routing-op path (patch 0013):
#   pin<n>   exactly n cold experts copied on every layer-token (TEMPORAL_SWAP_N)
#   one      exactly one per layer-token through the histogram path (TEMPORAL_SWAP_HIST=0,1)
#   skliar   the count drawn per layer-token from Skliar's histogram on the 1.26 B model at matched loads
# each under  s2 (TEMPORAL_ROUTE_SLOTS=2, switch off), s2f (the same with TEMPORAL_UNIFIED_FUSE=1) and
# lf1 (TEMPORAL_LOAD_FIRST=1). Timing only: the experts copied are not the ones computed.
# Protocol of PRETRAINED_LEDGER P1-12: llama-bench -ngl 99 -fa 1 -ub 1 -b 1 -d 1024 -n 128 -r 8, streamed arms
# with -ncmoe 48 TEMPORAL_UNIFIED=1; here every routed arm also has LLAMA_TEMPORAL_ROUTE=decode (the routing op
# the switch needs). Passes alternate the arm order.
#   big11b.sh <llama.cpp-dir> <model.gguf> <out-dir> <skliar histogram "w0,w1,..."> [passes]
set -uo pipefail
BIN="$1/build/bin/llama-bench"; M="$2"; OUT="$3"; SK="$4"; PASSES="${5:-3}"
mkdir -p "$OUT"
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
run() {   # tag, ncmoe flag, env...
  local tag="$1" nc="$2"; shift 2
  ( nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$tag.vram" & echo $! > "$OUT/smi.pid" )
  env "$@" taskset -c $CPUS "$BIN" -m "$M" -ngl 99 -fa 1 $nc -ub 1 -b 1 -d 1024 -n 128 -r 8 -o json > "$OUT/$tag.json" 2> "$OUT/$tag.err"
  kill "$(cat "$OUT/smi.pid")" 2>/dev/null
  python3 - "$OUT" "$tag" <<'PY'
import json, sys, re
o, t = sys.argv[1:]
try:
    r = json.load(open(f"{o}/{t}.json"))[-1]
    h = re.findall(r"temporal-swap-hist: (.*)", open(f"{o}/{t}.err").read())
    v = max(int(x) for x in open(f"{o}/{t}.vram").read().split() or [0])
    print(f"{t}: {r['avg_ts']:.2f} tok/s (sd {r['stddev_ts']:.2f}) peak {v} MiB {h[-1] if h else ''}", flush=True)
except Exception as e:
    print(f"{t}: FAILED {e}", flush=True)
PY
}
ST="TEMPORAL_UNIFIED=1 LLAMA_TEMPORAL_ROUTE=decode"
L=("ceiling||" "ceiling-route||LLAMA_TEMPORAL_ROUTE=decode")
for cfg in "s2|TEMPORAL_ROUTE_SLOTS=2" "s2f|TEMPORAL_ROUTE_SLOTS=2 TEMPORAL_UNIFIED_FUSE=1" "lf1|TEMPORAL_LOAD_FIRST=1"; do
  c=${cfg%%|*}; X=${cfg#*|}
  for n in 0 1 2 3; do L+=("pin$n-$c|-ncmoe 48|$ST $X TEMPORAL_SWAP_N=$n"); done
  L+=("one-$c|-ncmoe 48|$ST $X TEMPORAL_SWAP_HIST=0,1" "skliar-$c|-ncmoe 48|$ST $X TEMPORAL_SWAP_HIST=$SK")
done
for p in $(seq 1 "$PASSES"); do
  ORDER=$(seq 0 $((${#L[@]} - 1))); [ $((p % 2)) = 0 ] && ORDER=$(seq $((${#L[@]} - 1)) -1 0)
  for i in $ORDER; do
    IFS='|' read -r tag nc E <<< "${L[$i]}"
    run "p$p-$tag" "$nc" $E
  done
  echo "### pass $p done $(date -u +%T)"
done
echo RUN_DONE
