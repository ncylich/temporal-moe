#!/usr/bin/env bash
# 11 B random-weight Qwen3-MoE (fine, 18 of 192, 45 layers) on one GPU: decode speed when each layer-token's
# number of expert loads follows Skliar's measured histogram (TEMPORAL_SWAP_HIST, engine patch 0012) against
# exactly one load per layer-token on the same kernels. Reference rows: all experts in VRAM, zero loads,
# the paper's deploy path, and pinned n loads per layer.
#   run_hist.sh <llama.cpp-dir> <model.gguf> <out-dir> [passes]
set -uo pipefail
BIN="$1/build/bin/llama-bench"; M="$2"; OUT="$3"; PASSES="${4:-3}"
# Skliar on moe_fine_g3_1e19, lambda 0.7207, five real texts, 26,000 decoded token-layers (PRETRAINED_LEDGER P1-9)
SK="5047,16440,3074,1008,292,93,35,10,1"
mkdir -p "$OUT"
run() {   # tag, ncmoe flag, env...
  local tag="$1" nc="$2"; shift 2
  ( nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$tag.vram" & echo $! > "$OUT/smi.pid" )
  env TEMPORAL_UNIFIED_OVERLAP=1 "$@" "$BIN" -m "$M" -ngl 99 -fa 1 $nc -ub 1 -b 1 -d 1024 -n 128 -r 8 -o json \
      > "$OUT/$tag.json" 2> "$OUT/$tag.err"
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
for p in $(seq 1 "$PASSES"); do
  run "p$p-ceiling" ""
  run "p$p-n0"      "-ncmoe 48" TEMPORAL_UNIFIED=1 TEMPORAL_SWAP_N=0
  run "p$p-one"     "-ncmoe 48" TEMPORAL_UNIFIED=1 TEMPORAL_SWAP_HIST=0,1
  run "p$p-skliar"  "-ncmoe 48" TEMPORAL_UNIFIED=1 TEMPORAL_SWAP_HIST=$SK
  run "p$p-one-b"   "-ncmoe 48" TEMPORAL_UNIFIED=1 TEMPORAL_SWAP_HIST=0,1
  run "p$p-skliar-b" "-ncmoe 48" TEMPORAL_UNIFIED=1 TEMPORAL_SWAP_HIST=$SK
  run "p$p-deploy"  "-ncmoe 48" TEMPORAL_UNIFIED=1 TEMPORAL_SWAP_PROB=1.0
  for n in 1 2 3 4; do run "p$p-pin$n" "-ncmoe 48" TEMPORAL_UNIFIED=1 TEMPORAL_SWAP_N=$n; done
done
echo RUN_DONE
