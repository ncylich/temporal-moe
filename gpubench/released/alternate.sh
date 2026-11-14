#!/usr/bin/env bash
# Alternate arms of run_released.py N times (ABAB...), each a separate llama-bench process with the
# standard protocol, appending rows to one file. Used when run-to-run spread is larger than arm gaps.
#   alternate.sh <N> <out.jsonl> <tag> <ctx> <arm> [<arm> ...]
set -uo pipefail
N=$1; OUT=$2; TAG=$3; CTX=$4; shift 4
cd /workspace/rel
for i in $(seq 1 "$N"); do
  for arm in "$@"; do
    python3 repo/gpubench/released/run_released.py --bin eng/llama.cpp/build/bin --model ${MODEL:-/root/m/qwen35-q4km.gguf} \
      --nmoe ${NMOE:-40} --C 8 --lam ${SK_LAM:-0.4} --tokens ${TOK:-/workspace/rel/tok_qwen} --ctx "$CTX" --arms "$arm" \
      --out "$OUT" --tag "$TAG" --note "alternation $i" ${PREFIX:+--prefix "$PREFIX"} | cut -c1-110
  done
done
echo ALT_DONE
