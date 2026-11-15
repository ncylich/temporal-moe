#!/bin/sh
# Timing order used for PIXEL_SKLIAR_LEDGER.md. Run from androidbench/. Every run waits for the cool gate.
# 1. the trained model's deploy arm exactly as in TRAINED_MODEL_LEDGER.md step 6, on this build (check row)
python3 run_pixel_skliar.py time --tag t2 --texts 0 --arms trained_deploy --ngen 128 --reps 3
# 2. Skliar on the plain MoE, pool on flash, five real texts, context filled once per run as in that ledger
python3 run_pixel_skliar.py time --tag t2 --texts 1 2 3 4 5 --arms skliar_stream --ngen 128 --reps 3 --env TEMPORAL_BENCH_REPREFILL=0
echo TIMING_DONE
