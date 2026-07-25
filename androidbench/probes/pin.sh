#!/system/bin/sh
# pin scaling_min_freq to the current max so DVFS cannot drop cores during fetch stalls
for c in 0 4 7; do
  D=/sys/devices/system/cpu/cpu$c/cpufreq
  M=$(cat $D/scaling_max_freq)
  echo $M > $D/scaling_min_freq 2>/dev/null
done
for c in 0 4 7; do
  D=/sys/devices/system/cpu/cpu$c/cpufreq
  echo "cpu$c min=$(cat $D/scaling_min_freq) max=$(cat $D/scaling_max_freq) cur=$(cat $D/scaling_cur_freq)"
done
