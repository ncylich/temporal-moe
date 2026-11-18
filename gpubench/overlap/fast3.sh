#!/usr/bin/env bash
# The iteration loop for the fused op on the 11 B random-weight model (uniform Q4_0 file): three arms, about
# ten seconds of timed decode each, nothing else.
#   one    ours, streamed (18 slots), our routing kernel, exactly one load per layer-token
#   sk     Skliar, streamed, k_skliar executing, loads drawn from its histogram on the 1.26 B model
#   vram   ours, every expert resident
# Decode is what PRETRAINED_LEDGER P1-12 times (one token per step at depth 1,024, flash attention on). Two
# things differ from P1-12 to make the loop short, neither inside the timed region: the 1,024-token depth
# fill runs as one batch (FILL=1 restores the batch-1 fill), and each arm is REPS repetitions of 128 tokens
# (default 16, about 9 s) instead of 8. llama-bench fills the depth once per process and restores the saved
# state before each later repetition, so the fill is paid once per arm either way.
#   fast3.sh <llama.cpp-dir> <model.gguf> [tag]      env: REPS, FILL, ARMS="one sk vram", REV=1, X="extra env"
# Prints tok/s, ms per token and wall seconds per arm. REV=1 runs the arms in reverse order.
set -uo pipefail
BIN="$1/build/bin/llama-bench"; M="$2"; TAG="${3:-try}"; REPS="${REPS:-16}"; FILL="${FILL:-1024}"
SK="${SK:-16413,50220,7956,2391,732,186,96,6}"; LAM="${LAM:-0.739453125}"
OUT="${OUT:-/workspace/rel/ov/fast}"; mkdir -p "$OUT"
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
ARMS="${ARMS:-one sk vram}"; [ "${REV:-0}" = 1 ] && ARMS=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')
T0=$(date +%s.%N)
for a in $ARMS; do
  case $a in
    one)  E="TEMPORAL_UNIFIED=1 LLAMA_TEMPORAL_ROUTE=decode TEMPORAL_SWAP_HIST=0,1"; NC="-ncmoe 48";;
    sk)   E="TEMPORAL_UNIFIED=1 LLAMA_TEMPORAL_ROUTE=skliar TEMPORAL_CB_C=18 TEMPORAL_CB_J=1 TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_SWAP_HIST=$SK"; NC="-ncmoe 48";;
    vram) E="LLAMA_TEMPORAL_ROUTE=decode"; NC="";;
    *) echo "unknown arm $a"; exit 1;;
  esac
  t0=$(date +%s.%N)
  env TEMPORAL_FUSED=${FUSED:-1} $E ${X:-} taskset -c $CPUS "$BIN" -m "$M" -ngl 99 -fa 1 $NC -ub $FILL -b $FILL -d 1024 -n 128 -r $REPS -o json > "$OUT/$TAG-$a.json" 2> "$OUT/$TAG-$a.err"
  python3 - "$OUT/$TAG-$a" "$TAG" "$a" "$t0" <<'PY'
import json, sys, re, time
f, tag, a, t0 = sys.argv[1:]
try:
    r = json.load(open(f + ".json"))[-1]
    s = sorted(r["samples_ts"]); med = s[len(s)//2]
    e = open(f + ".err").read(); h = re.findall(r"temporal-swap-hist: layer_tokens=(\d+) loads=(\d+)", e)
    ld = f" loads {int(h[-1][1])/int(h[-1][0]):.4f}" if h else ""
    print(f"{tag} {a:5s} {r['avg_ts']:7.2f} tok/s  {1000/r['avg_ts']:.4f} ms  (median {med:.2f}, sd {r['stddev_ts']:.2f}){ld}  wall {time.time()-float(t0):.0f}s", flush=True)
except Exception as ex:
    print(f"{tag} {a}: FAILED {ex}", flush=True)
PY
done
echo "$TAG total wall $(python3 -c "import time;print(round(time.time()-$T0))")s fill=$FILL reps=$REPS"
