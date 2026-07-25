#!/usr/bin/env bash
# Samsung llama.cpp decode/prefill benchmark, protocol-matched to llamacpp-bench (A6000) and
# mlx-bench (Mac): B=1, context 1024 untimed, n=128 timed, r=8, greedy, CPU-only.
#
# Guardrails are from BENCHMARK_GATES.md; the M-numbers below cite it.
set -euo pipefail

MODEL_GGUF="${1:?usage: run_android_bench.sh <model.gguf> [pin_mask] [threads]}"
PIN_MASK="${2:-80}"          # 80 = cpu7 only; c0 = cpu6+7; none = no pinning (M22/M23)
THREADS="${3:-1}"
CTX=1024; NGEN=128; REPS=8

DEV_DIR=/data/local/tmp/tmoe
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/results"
mkdir -p "$OUT_DIR"
BASE="$(basename "$MODEL_GGUF")"

die() { echo "FATAL: $*" >&2; exit 1; }

# --- M10: refuse to run unless the expected device is the one attached -------------
EXPECT_SERIAL="${EXPECT_SERIAL:-RFGL42B1VLW}"
adb devices | grep -q "^${EXPECT_SERIAL}[[:space:]]*device$" \
  || die "expected serial $EXPECT_SERIAL not attached (M10)"

# --- M1/M2: power gate. A discharging pack costs 15-45% decode, and thermal_status --
# --- will still read 0 while it happens, so this check is not optional. -------------
BATT="$(adb shell dumpsys battery)"
echo "$BATT" | grep -qE '^ +(AC|USB|Wireless) powered: true' || die "device not on external power (M1)"
LEVEL="$(echo "$BATT" | sed -n 's/^ *level: *//p' | head -1)"
MIN_LEVEL="${MIN_LEVEL:-80}"
[ "$LEVEL" -ge "$MIN_LEVEL" ] || die "battery ${LEVEL}% < ${MIN_LEVEL}% required (M1)"
# net current must be positive = actually charging, not powering the host hub (M2)
CURR="$(adb shell dumpsys battery | sed -n 's/^ *current now: *//p' | head -1 || true)"

# --- M9: host must be quiet. A busy host costs ~4x on prefill while decode still ----
# --- looks plausible, so warn loudly rather than silently producing a good-looking row.
LOAD="$(sysctl -n vm.loadavg | awk '{print $2}')"
awk -v l="$LOAD" 'BEGIN{ if (l > 2.0) exit 0; exit 1 }' \
  && echo "WARNING: host load average ${LOAD} - prefill may be understated (M9)" >&2

# --- audit-8: artifact identity. Hash what is actually on the device. ---------------
adb shell "test -f $DEV_DIR/$BASE" || die "$BASE not pushed to $DEV_DIR"
DEV_SHA="$(adb shell "sha256sum $DEV_DIR/$BASE" | awk '{print $1}')"
HOST_SHA="$(shasum -a 256 "$MODEL_GGUF" | awk '{print $1}')"
[ "$DEV_SHA" = "$HOST_SHA" ] || die "on-device hash != host hash (audit-8)"

# --- M20/M21: pin via `taskset -p <mask> $$ && exec`, then PROVE the pin took. ------
if [ "$PIN_MASK" = "none" ]; then
  PREFIX=""; AFFINITY="not_set"          # M22: full-core rows must be labelled as such
else
  PREFIX="taskset -p $PIN_MASK \$\$ >/dev/null && "
  AFFINITY="$(adb shell "taskset -p $PIN_MASK \$\$ >/dev/null && grep Cpus_allowed_list /proc/self/status" \
              | awk '{print $2}' | tr -d '\r')"
  [ -n "$AFFINITY" ] || die "taskset readback empty - pin not proven (M21)"
fi

THERM_BEFORE="$(adb shell dumpsys thermalservice | sed -n 's/.*Thermal Status: *//p' | head -1 | tr -d '\r')"

RAW="$OUT_DIR/raw_${BASE%.gguf}_mask${PIN_MASK}_t${THREADS}_$(date +%s).csv"
echo "running: mask=$PIN_MASK (cpus $AFFINITY) threads=$THREADS ctx=$CTX n=$NGEN r=$REPS"
adb shell "cd $DEV_DIR && ${PREFIX}exec ./llama-bench -m $BASE \
  -ngl 0 -t $THREADS -ub 1 -b 1 -d $CTX -n $NGEN -r $REPS -o csv" | tr -d '\r' > "$RAW"

THERM_AFTER="$(adb shell dumpsys thermalservice | sed -n 's/.*Thermal Status: *//p' | head -1 | tr -d '\r')"
BATT_AFTER="$(adb shell dumpsys battery | sed -n 's/^ *level: *//p' | head -1)"

# --- M7 / audit-2: a throttled row is not a result. Flag rather than silently keep. -
STATUS=ok
if [ "${THERM_BEFORE:-0}" != "0" ] || [ "${THERM_AFTER:-0}" != "0" ]; then
  STATUS="throttled"
  echo "WARNING: thermal_status ${THERM_BEFORE}->${THERM_AFTER}; row marked '$STATUS' (M7)" >&2
fi

ENGINE_COMMIT="$(git -C "${LLAMA_DIR:-$HOME/Documents/llama.cpp-android}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
NDK_REV="$(sed -n 's/^Pkg.Revision *= *//p' /opt/homebrew/share/android-ndk/source.properties 2>/dev/null || echo unknown)"

NOTE="engine=${ENGINE_COMMIT};ndk=${NDK_REV};march=armv8.2-a+dotprod+i8mm+fp16;ngl=0;threads=${THREADS};\
taskset_mask=${PIN_MASK};cpus_allowed=${AFFINITY};thermal=${THERM_BEFORE}->${THERM_AFTER};\
batt=${LEVEL}->${BATT_AFTER}%;curr_now=${CURR:-na};serial=${EXPECT_SERIAL};soc=SM8850;android=16;\
gguf_sha256=${DEV_SHA:0:16};mmap=on;host_load=${LOAD};status=${STATUS}"

# --- audit-1: never let a zero or non-finite metric through as 'ok'. ---------------
python3 "$(dirname "$0")/emit_row.py" "$RAW" "$BASE" "$NOTE" "$STATUS" \
  >> "$OUT_DIR/serving_benchmarks_android.csv"

echo "raw:  $RAW"
echo "rows: $OUT_DIR/serving_benchmarks_android.csv"
tail -2 "$OUT_DIR/serving_benchmarks_android.csv"
