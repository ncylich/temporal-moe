#!/bin/sh
# Timed rows for ledger S1-7: LLAMA_TEMPORAL_SHEXP_FIRST off (tag sx0) against on (tag sx1), same binary.
# Run from androidbench/. Every run waits for the cool gate and rests 60 s. Per text, the three arms in turn,
# each arm off and on back to back; the off/on order flips on every other text.
export LAM=0.7310
E="TEMPORAL_BENCH_REPREFILL=0"
for v in 1 2 3 4 5; do
  for arm in trainedtext_stream skliar_stream stock_resident; do
    if [ $((v % 2)) -eq 1 ]; then order="0 1"; else order="1 0"; fi
    for s in $order; do
      python3 run_pixel_skliar.py time --tag sx$s --texts $v --arms $arm --ngen 128 --reps 3 --env $E LLAMA_TEMPORAL_SHEXP_FIRST=$s
    done
  done
done
echo TIMING_DONE
