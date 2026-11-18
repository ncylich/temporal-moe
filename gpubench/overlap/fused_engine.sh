#!/usr/bin/env bash
# Engine patch 0014 (TEMPORAL_FUSED) on the 1e19 fine models: gates, then decode speed under the P1-9 protocol.
# Gates: batch-1 perplexity (4 x 1,024 tokens) for our rule and Skliar on the plain MoE and for the trained model,
# all in VRAM and streamed, with the fused op off and on. Fused streamed must equal fused all-in-VRAM in every
# digit; fused against unfused is reported (sums are formed in a different order, see the patch).
# Timing: as gpubench/overlap/speed.sh (rows 601-605, four trials, first dropped, passes alternate the order).
#   fused_engine.sh <lambda> <passes> [outdir] [gates|speed|both]
set -uo pipefail
LAM=$1; NP=$2; O=${3:-/workspace/rel/ov/fused}; WHAT=${4:-both}; BIN=/workspace/rel/eng/llama.cpp/build/bin
cd /workspace/rel; R=repo/gpubench; mkdir -p $O
MOE=m/moe_fine_1e19-full-q4_0.gguf; TM=m/fine_1e19-q4_0.gguf; T=m/test_tokens_2048.npz
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
if [ $WHAT != speed ]; then
  echo "### gates $(date -u +%T)"
  PPL() { env "$@" > $O/last.log 2>&1; echo "$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/last.log || echo FAILED) | $(grep -E '^temporal-(route2|route-cuda|cuda)|temporal-fused' $O/last.log | cut -c1-110 | tr '\n' '|')"; }
  for mode in decode skliar trained; do
    M=$MOE; [ $mode = trained ] && M=$TM
    E="LLAMA_TEMPORAL_ROUTE=$mode TEMPORAL_CB_C=18 TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_J=1"
    C="$BIN/llama-perplexity -m $M -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa off -t 8"
    echo "$mode vram unfused: $(PPL $E $C)"
    echo "$mode vram fused:   $(PPL $E TEMPORAL_FUSED=1 $C)"
    echo "$mode stream unfused (slots 2): $(PPL $E TEMPORAL_UNIFIED=1 TEMPORAL_ROUTE_SLOTS=2 $C -ncmoe 14)"
    for rep in 1 2; do echo "$mode stream fused rep $rep: $(PPL $E TEMPORAL_UNIFIED=1 TEMPORAL_FUSED=1 $C -ncmoe 14)"; done
    for rep in 1 2; do echo "$mode stream fused mode 2 rep $rep: $(PPL $E TEMPORAL_UNIFIED=1 TEMPORAL_FUSED=2 $C -ncmoe 14)"; done
  done
  echo "### routing equality, fused streamed $(date -u +%T)"
  stop() { [ -f "$1.pid" ] || return 0; kill -INT "$(cat "$1.pid")" 2>/dev/null; for i in $(seq 1 60); do kill -0 "$(cat "$1.pid")" 2>/dev/null || { rm -f "$1.pid"; return 0; }; sleep 1; done; kill -9 "$(cat "$1.pid")" 2>/dev/null; rm -f "$1.pid"; }
  export SK_C=18 SK_LAM=$LAM NMOE=14
  for arm in ours_stream skliar_stream; do
    mode=${arm%_stream}; [ "$mode" = ours ] && mode=decode
    bash $R/released/server.sh $arm $MOE 8091 $O/srv_$arm.log GGML_CUDA_DISABLE_GRAPHS=1 TEMPORAL_FUSED=1 LLAMA_TEMPORAL_ROUTE_DUMP2=$O/route_$arm.bin > /dev/null || continue
    python3 $R/pretrained/tokens_base.py --url http://127.0.0.1:8091 --tokens $T --row 700 --ctx 512 --tag dump --out $O/dump_$arm > /dev/null
    stop $O/srv_$arm.log
    echo "fused $arm: $(python3 $R/released/route_check.py --dump $O/route_$arm.bin --mode $mode --K 18 --E 192 --C 18 --lam $LAM --out $O/check_fused_$arm.json | tail -1 | cut -c1-160)"
    rm -f $O/route_$arm.bin
  done
  echo "### GATES_DONE $(date -u +%T)"
fi
[ $WHAT = gates ] && exit 0
for v in 1 2 3 4 5; do for m in base ours skliar trained; do
  python3 $R/pretrained/tokens_base.py --tokens $T --row $((600 + v)) --ctx 1024 --tag $m --out $O/tok_v$v > /dev/null
done; done
ARMS=("vram-stock|$MOE|base|")
for a in "ours|$MOE|ours" "skliar|$MOE|skliar" "trained|$TM|trained"; do
  IFS='|' read -r n M arm <<< "$a"
  ARMS+=(
   "vram-$n|$M|${arm}_vram|"
   "vram-$n-fused|$M|${arm}_vram|TEMPORAL_FUSED=1"
   "$n-s2|$M|${arm}_stream|TEMPORAL_ROUTE_SLOTS=2"
   "$n-lf3f|$M|${arm}_stream|TEMPORAL_LOAD_FIRST=3 TEMPORAL_UNIFIED_FUSE=1"
   "$n-fused|$M|${arm}_stream|TEMPORAL_FUSED=1"
   "$n-fused-nb|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_ROUTE_NOCOPY=1"
   "$n-fused-skipA|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_ROUTE_NOCOPY=1 TEMPORAL_FUSED_SKIP=1"
   "$n-fused-skipB|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_ROUTE_NOCOPY=1 TEMPORAL_FUSED_SKIP=2"
   "$n-fused-skipAB|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_ROUTE_NOCOPY=1 TEMPORAL_FUSED_SKIP=3"
   "$n-fused-stub|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_ROUTE_STUB=1"
   "vram-$n-fused-stub|$M|${arm}_vram|TEMPORAL_FUSED=1 TEMPORAL_ROUTE_STUB=1"
   "$n-fused2|$M|${arm}_stream|TEMPORAL_FUSED=2"
   "$n-fused2-nb|$M|${arm}_stream|TEMPORAL_FUSED=2 TEMPORAL_ROUTE_NOCOPY=1"
   "$n-f1-pin0|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_SWAP_N=0"
   "$n-f1-pin1|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_SWAP_N=1"
   "$n-f1-pin2|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_SWAP_N=2"
   "$n-f1-pin3|$M|${arm}_stream|TEMPORAL_FUSED=1 TEMPORAL_SWAP_N=3"
   "$n-f2-pin0|$M|${arm}_stream|TEMPORAL_FUSED=2 TEMPORAL_SWAP_N=0"
   "$n-f2-pin1|$M|${arm}_stream|TEMPORAL_FUSED=2 TEMPORAL_SWAP_N=1"
   "$n-f2-pin2|$M|${arm}_stream|TEMPORAL_FUSED=2 TEMPORAL_SWAP_N=2"
   "$n-f2-pin3|$M|${arm}_stream|TEMPORAL_FUSED=2 TEMPORAL_SWAP_N=3"
  )
done
for p in $(seq 1 $NP); do
  ORDER=$(seq 0 $((${#ARMS[@]} - 1))); [ $((p % 2)) = 0 ] && ORDER=$(seq $((${#ARMS[@]} - 1)) -1 0)
  for v in 1 2 3 4 5; do for i in $ORDER; do
    IFS='|' read -r label M arm X <<< "${ARMS[$i]}"
    [ -n "${ONLY:-}" ] && ! [[ $label =~ $ONLY ]] && continue
    printf "%-20s p%d v%d " $label $p $v
    python3 $R/released/run_released.py --bin $BIN --nmoe 14 --C 18 --lam $LAM --tokens $O/tok_v$v --out $O/${OUTFILE:-speed_fused.jsonl} --fa 0 \
       --prefix "taskset -c $CPUS" --ctx 1024 --model $M --tag $label --note "pass $p row $((600 + v))" --env TEMPORAL_SHEXP_FIRST=1 $X --arms $arm | cut -c16-120
  done; done
  echo "### pass $p done $(date -u +%T)"
done
echo "### SPEED_DONE $(date -u +%T)"
