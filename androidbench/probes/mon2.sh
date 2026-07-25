#!/system/bin/sh
U=/sys/devices/platform/13200000.ufs/monitor
CFG="$1"; CHUNK="$2"; shift 2
echo 0 > $U/monitor_enable
echo $CHUNK > $U/monitor_chunk_size
echo 1 > $U/monitor_enable
cd /data/local/tmp/tmoe
echo 1000 > /proc/self/oom_score_adj
env $@ ./llama-bench-temporal -m qwen3moe-rand-e112-Q4pure.gguf -t 4 -p 0 -n 32 -r 1 -mmp 0 -ot "_exps=CPU" -o csv 2>&1 | grep -E "fetchprof" | cut -c1-110
N=$(cat $U/read_nr_requests); A=$(cat $U/read_req_latency_avg); MX=$(cat $U/read_req_latency_max)
MN=$(cat $U/read_req_latency_min); B=$(cat $U/read_total_busy); SEC=$(cat $U/read_total_sectors)
echo 0 > $U/monitor_enable
echo "  UFS-DRIVER $CFG (chunk=$CHUNK): nr=$N  avg=$((A/1000))us  min=$((MN/1000))us  max=$((MX/1000))us  busy=$((B/1000000))ms  MiB=$((SEC/2048))"
