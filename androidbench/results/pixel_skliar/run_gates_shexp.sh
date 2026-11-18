#!/bin/sh
# Gates for ledger S1-7 (LLAMA_TEMPORAL_SHEXP_FIRST). Run from androidbench/. No cool gate: only hashes and routing are read.
export LAM=0.7310
S="LLAMA_TEMPORAL_SHEXP_FIRST=1"
# Skliar, switch on: routing against the Python reference, flash == RAM, flash repeat
python3 run_pixel_skliar.py gate --tag g3on --texts 1 2 3 4 5 --reps 3 --env TEMPORAL_BENCH_REPREFILL=0 $S
# Skliar, switch off, pool on flash: hashes to compare with the switch-on runs
python3 run_pixel_skliar.py quick --tag g3off --arms skliar_stream --texts 1 2 3 4 5 --reps 3 --env TEMPORAL_BENCH_REPREFILL=0 TEMPORAL_BENCH_LOGIT_HASH=1
# trained model on the same texts: flash on, RAM on, flash off
python3 run_pixel_skliar.py quick --tag g3on --arms trainedtext_stream trainedtext_ram2p --texts 1 2 3 4 5 --reps 3 --env TEMPORAL_BENCH_REPREFILL=0 TEMPORAL_BENCH_LOGIT_HASH=1 $S
python3 run_pixel_skliar.py quick --tag g3off --arms trainedtext_stream --texts 1 2 3 4 5 --reps 3 --env TEMPORAL_BENCH_REPREFILL=0 TEMPORAL_BENCH_LOGIT_HASH=1
echo GATES_DONE
