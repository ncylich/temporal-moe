#!/usr/bin/env bash
# Decode serving benchmark runner for the temporal-MoE llama.cpp fork.
# Emits one CSV-ready line (tok/s + peak VRAM) per invocation. Protocol matches the
# existing decode rows: B=1, 1024-token context depth, n=128 decode tokens, -r 8, -fa 1.
#
# Usage: run_serving_bench.sh <bin-dir> <model.gguf> <setup> [extra llama-bench args...]
#   setup selects env:
#     ceiling  : all experts resident (no -ncmoe)                 -> row a
#     deploy   : TEMPORAL_UNIFIED, <=1 swap/layer, SWAP_PROB=1.0   -> row c
#     floor    : TEMPORAL_UNIFIED + NOFORCE1 (budget=R multiswap)  -> vanilla LRU-on-miss
#     floor_budget=<n> : floor with pinned per-layer swap budget n (needs SWAP_BUDGET knob)
set -euo pipefail
BIN="${1:?bin dir}"; MODEL="${2:?model gguf}"; SETUP="${3:?setup}"; shift 3
NCMOE=48
COMMON=(-m "$MODEL" -ngl 99 -fa 1 -ub 1 -b 1 -d 1024 -n 128 -r 8 -o csv)

ENV=()
case "$SETUP" in
  ceiling) EXTRA=() ;;
  deploy)  ENV=(TEMPORAL_UNIFIED=1 TEMPORAL_UNIFIED_OVERLAP=1 TEMPORAL_SWAP_PROB=1.0); EXTRA=(-ncmoe $NCMOE) ;;
  floor)   ENV=(TEMPORAL_UNIFIED=1 TEMPORAL_UNIFIED_NOFORCE1=1);                        EXTRA=(-ncmoe $NCMOE) ;;
  floor_n=*) n="${SETUP#floor_n=}"; ENV=(TEMPORAL_UNIFIED=1 TEMPORAL_SWAP_N=$n); EXTRA=(-ncmoe $NCMOE) ;;   # pinned n swaps/layer (emulated floor)
  *) echo "unknown setup $SETUP" >&2; exit 2 ;;
esac

# poll peak VRAM (global) during the run
VRAMF=$(mktemp)
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done ) > "$VRAMF" &
POLL=$!

OUT=$(env "${ENV[@]}" "$BIN/llama-bench" "${COMMON[@]}" "${EXTRA[@]}" "$@" 2>/tmp/bench_err.log) || { kill $POLL; cat /tmp/bench_err.log >&2; exit 1; }
kill $POLL 2>/dev/null || true

# avg_ts is the token-gen throughput column in llama-bench csv
TS=$(echo "$OUT" | awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="avg_ts") c=i} NR==2{gsub(/"/,"",$c); print $c}')
PEAK=$(sort -n "$VRAMF" | tail -1); rm -f "$VRAMF"
echo "SETUP=$SETUP tok_s=$TS peak_vram_mib=$PEAK"
echo "$OUT" | grep -i "temporal\|UNIFIED" >&2 || true
