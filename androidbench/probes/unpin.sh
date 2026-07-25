#!/system/bin/sh
echo 820000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null
echo 357000 > /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq 2>/dev/null
echo 700000 > /sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq 2>/dev/null
for c in 0 4 7; do echo "cpu$c min=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq)"; done
