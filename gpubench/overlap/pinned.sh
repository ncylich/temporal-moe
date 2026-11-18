#!/usr/bin/env bash
# Pinned loads per layer-token (TEMPORAL_SWAP_N=n, n cold experts copied whatever the router selects; timing
# only, output wrong), switch off (TEMPORAL_ROUTE_SLOTS=2) against TEMPORAL_LOAD_FIRST=3 with fusion and
# TEMPORAL_LOAD_FIRST=1, on one model. Shows
# what the first load of a layer costs against a later one. Same protocol as the timed runs; passes alternate.
#   pinned.sh <model> <ncmoe> <fa> <arm> <tokens dir> <passes> <out.jsonl> [n list]
set -uo pipefail
M=$1; NM=$2; FA=$3; ARM=$4; TD=$5; NP=$6; OUT=$7; NS=${8:-"0 1 2 3"}; BIN=/workspace/rel/eng/llama.cpp/build/bin
cd /workspace/rel; R=repo/gpubench
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
L=(); for n in $NS; do L+=("pin$n-s2|TEMPORAL_ROUTE_SLOTS=2 TEMPORAL_SWAP_N=$n" "pin$n-lf3f|TEMPORAL_LOAD_FIRST=3 TEMPORAL_UNIFIED_FUSE=1 TEMPORAL_SWAP_N=$n" "pin$n-lf1|TEMPORAL_LOAD_FIRST=1 TEMPORAL_SWAP_N=$n"); done
for p in $(seq 1 $NP); do
  ORDER=$(seq 0 $((${#L[@]} - 1))); [ $((p % 2)) = 0 ] && ORDER=$(seq $((${#L[@]} - 1)) -1 0)
  for i in $ORDER; do
    IFS='|' read -r label X <<< "${L[$i]}"
    printf "%-12s p%d " $label $p
    python3 $R/released/run_released.py --bin $BIN --nmoe $NM --C ${SK_C:-18} --lam ${SK_LAM:-0.7} --tokens $TD --out $OUT --fa $FA \
       --prefix "taskset -c $CPUS" --ctx 1024 --model $M --tag $label --note "pass $p" --env TEMPORAL_SHEXP_FIRST=1 $X --arms $ARM | cut -c16-150
  done
done
echo "### PINNED_DONE $(date -u +%T)"
