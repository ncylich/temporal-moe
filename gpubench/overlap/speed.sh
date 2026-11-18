#!/usr/bin/env bash
# Decode speed with engine patch 0013's switch off and on, 1e19 fine models, RTX A6000 (pod layout
# /workspace/rel/{repo,eng,m}). Protocol of PRETRAINED_LEDGER P1-9: batch 1, Q4_0, -fa 0 -t 8, pinned to the GPU's
# NUMA node, 1,024-token prompt as one batch, the test row's own next 100 tokens decoded, rows 601-605, four
# trials with the first dropped, the same token files for every arm.
# Arms: all in VRAM (stock, our rule, Skliar on the plain MoE; the trained model), and for each of our rule
# (plain MoE), Skliar (plain MoE) and the trained model, streamed with
#   p19   the P1-9 setting (slots 2 for our rule, 1 for Skliar and the trained model)
#   s2    TEMPORAL_ROUTE_SLOTS=2, switch off          s2f   the same with TEMPORAL_UNIFIED_FUSE=1
#   lf3 / lf2 / lf1   TEMPORAL_LOAD_FIRST=3 / 2 / 1          lf3f  lf3 with TEMPORAL_UNIFIED_FUSE=1
#   nb-<setting>      that setting with TEMPORAL_ROUTE_NOCOPY=1 (copy kernels launched, no bytes moved; output wrong)
# ONLY=<regex> restricts a run to the matching labels; OUTFILE names the rows file (default speed.jsonl).
# Passes alternate the arm order (odd passes forward, even passes reversed).
#   speed.sh <lambda> <passes> [outdir]
set -uo pipefail
LAM=$1; NP=$2; O=${3:-/workspace/rel/ov/speed}; BIN=/workspace/rel/eng/llama.cpp/build/bin
cd /workspace/rel; R=repo/gpubench; mkdir -p $O
MOE=m/moe_fine_1e19-full-q4_0.gguf; TM=m/fine_1e19-q4_0.gguf; T=m/test_tokens_2048.npz
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
echo "### GPU0 CPU affinity: ${CPUS:-unknown} lambda $LAM $(date -u +%T)"
for v in 1 2 3 4 5; do for m in base ours skliar trained; do
  python3 $R/pretrained/tokens_base.py --tokens $T --row $((600 + v)) --ctx 1024 --tag $m --out $O/tok_v$v > /dev/null
done; done
# label|model|arm|extra env
ARMS=(
 "vram-stock|$MOE|base|"
 "vram-ours|$MOE|ours_vram|"
 "vram-skliar|$MOE|skliar_vram|"
 "vram-trained|$TM|trained_vram|"
)
for a in "ours|$MOE|ours_stream|2" "skliar|$MOE|skliar_stream|1" "trained|$TM|trained_stream|1"; do
  IFS='|' read -r n M arm s19 <<< "$a"
  ARMS+=(
   "$n-p19|$M|$arm|TEMPORAL_ROUTE_SLOTS=$s19"
   "$n-s2|$M|$arm|TEMPORAL_ROUTE_SLOTS=2"
   "$n-s2f|$M|$arm|TEMPORAL_ROUTE_SLOTS=2 TEMPORAL_UNIFIED_FUSE=1"
   "$n-lf3|$M|$arm|TEMPORAL_LOAD_FIRST=3"
   "$n-lf3f|$M|$arm|TEMPORAL_LOAD_FIRST=3 TEMPORAL_UNIFIED_FUSE=1"
   "$n-lf2|$M|$arm|TEMPORAL_LOAD_FIRST=2"
   "$n-lf1|$M|$arm|TEMPORAL_LOAD_FIRST=1"
   "$n-nb-s2|$M|$arm|TEMPORAL_ROUTE_SLOTS=2 TEMPORAL_ROUTE_NOCOPY=1"
   "$n-nb-lf1|$M|$arm|TEMPORAL_LOAD_FIRST=1 TEMPORAL_ROUTE_NOCOPY=1"
   "$n-nb-lf3|$M|$arm|TEMPORAL_LOAD_FIRST=3 TEMPORAL_ROUTE_NOCOPY=1"
   "$n-nb-lf3f|$M|$arm|TEMPORAL_LOAD_FIRST=3 TEMPORAL_UNIFIED_FUSE=1 TEMPORAL_ROUTE_NOCOPY=1"
   "$n-nb-s2f|$M|$arm|TEMPORAL_ROUTE_SLOTS=2 TEMPORAL_UNIFIED_FUSE=1 TEMPORAL_ROUTE_NOCOPY=1"
  )
done
for p in $(seq 1 $NP); do
  ORDER=$(seq 0 $((${#ARMS[@]} - 1))); [ $((p % 2)) = 0 ] && ORDER=$(seq $((${#ARMS[@]} - 1)) -1 0)
  for v in 1 2 3 4 5; do
    for i in $ORDER; do
      IFS='|' read -r label M arm X <<< "${ARMS[$i]}"
      [ -n "${ONLY:-}" ] && ! [[ $label =~ $ONLY ]] && continue
      printf "%-16s p%d v%d " $label $p $v
      python3 $R/released/run_released.py --bin $BIN --nmoe 14 --C 18 --lam $LAM --tokens $O/tok_v$v --out $O/${OUTFILE:-speed.jsonl} --fa 0 \
         --prefix "taskset -c $CPUS" --ctx 1024 --model $M --tag $label --note "pass $p row $((600 + v))" \
         --env TEMPORAL_SHEXP_FIRST=1 $X --arms $arm | cut -c16-160
    done
  done
  echo "### pass $p done $(date -u +%T)"
done
echo "### SPEED_DONE $(date -u +%T)"
