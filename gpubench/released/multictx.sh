#!/usr/bin/env bash
# Streamed arms timed on several different 1k conversations (variants 1..N), since Skliar's load rate, and
# with it its speed, depends on the text. Each variant: token files (tokens.sh, own continuation per
# method), then base, ours_stream and skliar_stream with TEMPORAL_SHEXP_FIRST=1 and the two streamed arms
# without it; standard protocol, pinned to the GPU's NUMA node.
#   multictx.sh <model.gguf> <nmoe> <lambda> <tokdir-prefix> <out.jsonl> <tag> <N>
set -uo pipefail
M=$1; NM=$2; LAM=$3; TP=$4; OUT=$5; TAG=$6; N=$7
cd /workspace/rel
for v in $(seq 1 "$N"); do
  NMOE=$NM VARIANT=$v bash repo/gpubench/released/tokens.sh "$M" "${TP}_v$v" 8 "$LAM" 1024 || continue
  R() { python3 repo/gpubench/released/run_released.py --bin eng/llama.cpp/build/bin --model "$M" --nmoe "$NM" --C 8 --lam "$LAM" \
        --tokens "${TP}_v$v" --ctx 1024 --out "$OUT" --prefix "taskset -c 0-31,64-95" --note "variant $v" "$@" | cut -c1-120; }
  R --tag "$TAG-shexp-v$v" --env TEMPORAL_SHEXP_FIRST=1 --arms base ours_stream skliar_stream
  R --tag "$TAG-default-v$v" --arms skliar_stream ours_stream
done
echo MULTICTX_DONE
