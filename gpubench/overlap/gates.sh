#!/usr/bin/env bash
# Gates for engine patch 0013 (TEMPORAL_LOAD_FIRST) on the 1e19 fine models, pod layout /workspace/rel/{repo,eng,m}.
#   1. batch-1 perplexity, streamed with the switch at 0 (slots 2) / 3 / 2 / 1, against all in VRAM: must be
#      identical in every digit for our rule and Skliar on the plain MoE and for the trained model;
#   2. routing equality against decode_state.py / cache_bias.py (route_check.py) with the switch at 1.
#   gates.sh <lambda> [outdir]
set -uo pipefail
LAM=$1; O=${2:-/workspace/rel/ov/gates}; export SK_C=18 SK_LAM=$LAM NMOE=14 BIN=/workspace/rel/eng/llama.cpp/build/bin
cd /workspace/rel; R=repo/gpubench; mkdir -p $O
MOE=m/moe_fine_1e19-full-q4_0.gguf; TM=m/fine_1e19-q4_0.gguf; T=m/test_tokens_2048.npz
stop() { [ -f "$1.pid" ] || return 0; kill -INT "$(cat "$1.pid")" 2>/dev/null; for i in $(seq 1 60); do kill -0 "$(cat "$1.pid")" 2>/dev/null || { rm -f "$1.pid"; return 0; }; sleep 1; done; kill -9 "$(cat "$1.pid")" 2>/dev/null; rm -f "$1.pid"; }
echo "### gate: batch-1 perplexity, 4 x 1024 tokens $(date -u +%T)"
echo "stock: $(env $BIN/llama-perplexity -m $MOE -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa off -t 8 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+')"
for mode in decode skliar trained; do
  M=$MOE; [ $mode = trained ] && M=$TM
  E="LLAMA_TEMPORAL_ROUTE=$mode TEMPORAL_CB_C=18 TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_J=1"
  echo "$mode vram: $(env $E $BIN/llama-perplexity -m $M -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa off -t 8 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+')"
  for LF in 0 3 2 1; do for rep in 1 2; do
    X="TEMPORAL_ROUTE_SLOTS=2"; [ $LF != 0 ] && X="TEMPORAL_LOAD_FIRST=$LF"
    env $E TEMPORAL_UNIFIED=1 TEMPORAL_SHEXP_FIRST=1 $X $BIN/llama-perplexity -m $M -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa off -t 8 -ncmoe 14 > $O/last.log 2>&1
    echo "$mode stream load_first $LF rep $rep: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/last.log || echo FAILED) | $(grep -E '^temporal-(route2|route-cuda|cuda)' $O/last.log | cut -c1-150 | tr '\n' '|')"
  done; done
done
echo "### gate: engine routing equals decode_state.py / cache_bias.py, switch at 1 $(date -u +%T)"
for arm in ours_stream skliar_stream; do
  mode=${arm%_stream}; [ "$mode" = ours ] && mode=decode
  bash $R/released/server.sh $arm $MOE 8091 $O/srv_dump_$arm.log GGML_CUDA_DISABLE_GRAPHS=1 TEMPORAL_LOAD_FIRST=1 TEMPORAL_SHEXP_FIRST=1 LLAMA_TEMPORAL_ROUTE_DUMP2=$O/route_$arm.bin > /dev/null || continue
  python3 $R/pretrained/tokens_base.py --url http://127.0.0.1:8091 --tokens $T --row 700 --ctx 512 --tag dump --out $O/dump_$arm > /dev/null
  stop $O/srv_dump_$arm.log
  echo "load_first 1 $arm: $(python3 $R/released/route_check.py --dump $O/route_$arm.bin --mode $mode --K 18 --E 192 --C 18 --lam $LAM --out $O/check_lf1_$arm.json | tail -1)"
  rm -f $O/route_$arm.bin
done
echo "### GATES_DONE $(date -u +%T)"
