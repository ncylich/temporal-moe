#!/bin/sh
# Timed rows for ledger S1-10: six reads per expert (LLAMA_TEMPORAL_SPLIT=2, tag sp2) against three (SPLIT=1,
# tag sp1). Shared expert first in every run, same binary, cool gate and 60 s rest. Two texts, two arms; the
# order of the two settings flips between texts. Run from androidbench/.
export LAM=0.7310
E="TEMPORAL_BENCH_REPREFILL=0 LLAMA_TEMPORAL_SHEXP_FIRST=1"
for v in 1 2; do
  for arm in trainedtext_stream skliar_stream; do
    if [ $v -eq 1 ]; then order="2 1"; else order="1 2"; fi
    for s in $order; do
      python3 run_pixel_skliar.py time --tag sp$s --texts $v --arms $arm --ngen 128 --reps 3 --env $E LLAMA_TEMPORAL_SPLIT=$s
    done
  done
done
echo TIMING_DONE
