#!/system/bin/sh
# usage: _mem.sh <depth> <madv_free 0|1>   prints RSS timeline summary of a deploy run
# Memory-only check for TRAINED_MODEL_LEDGER.md step 6 (not a timing run).
cd /data/local/tmp/tmoe
export LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE=fine_1e19-q4_0-repacked.bin LLAMA_TEMPORAL_ODIRECT=1
[ "$2" = 1 ] && export LLAMA_TEMPORAL_MADV_FREE=1
export LLAMA_TEMPORAL_R=18 LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=trained
D=""; [ "$1" != 0 ] && D="-d $1"
./llama-bench-trained -m fine_1e19-q4_0.gguf -t 4 -p 0 -n 256 $D -r 1 -mmp 0 -ot "_exps=CPU" -o csv > _mem.out 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  grep -E "VmRSS|RssAnon|RssFile|VmHWM" /proc/$P/status 2>/dev/null | tr -s ' \t' ' ' | tr '\n' ' '; echo
  sleep 0.25
done
wait $P
grep -E "temporal-pool: fetches|temporal-route" _mem.out
