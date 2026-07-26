# Benchmark gates

Rules the Android harness enforces mechanically. `run_android_bench.sh` and `bench.py`
tag themselves with these numbers so a failed gate points at the reason it exists.

Every one of these came from a run that produced believable numbers and was wrong.

## Device state

**M1. The device must be on AC or USB power, verified programmatically every run.**
A discharging pack current-limits the SoC and quietly costs 15 to 45% of decode.

**M2. Watch for the USB source-role trap, and do not trust `thermal_status=0`.**
"Charge connected device" puts the port in source role, so the phone powers the host hub
and discharges all session while reporting as connected. Check that net battery current is
positive rather than trusting the charging flag. Thermal status can read 0 while
`scaling_max_freq` sits far below the core's rating, so gate on observed clock, not on the
thermal counter.

**M4. Bench from rest, in short batches. One warmup, three measured rounds.**
Back-to-back hot batches read 20 to 30% lower.

**M5. Rest several minutes between batches.** A metal surface helps.

**M7. Record thermal status on every row and state the run condition.**
Cool and powered rows read 37.0 tok/s where back-to-back hot rows read 28.3, a 23% swing
from device state alone. A throttled row is not a result. Mark it rather than silently
keeping it.

## Host state

**M9. The host must be quiet too.** This is the nastiest failure mode in the set. A
concurrent compile on the host costs roughly 4x on compute-bound prefill while leaving
memory-bound decode looking entirely plausible. The run still produces believable numbers.

**M10. Pin adb device selection to a serial or exact model, and refuse to run otherwise.**
A two-phone bench can then never silently target the wrong handset. Set `ANDROID_SERIAL`
for your own device.

## Core pinning

**M20. Pin with `taskset -p <mask> $$ && exec <runner>`, not `taskset <mask> <runner>`.**
Retargeting the shell's own PID and then exec-ing means every thread the runtime spawns
inherits the mask and cannot widen back out.

**M21. Prove the pin took. Do not assume it.** The positive control is
`taskset 80 sh -c 'grep Cpus_allowed_list /proc/self/status'`, which must print `7`.
Capture the readback into the row.

**M22. Single-prime-core can beat all-core for small models and lose for larger ones.
Always report which.** At the 512+32 spec on a Pixel 10a, qwen3-0.6B decodes about 26 tok/s
single-prime against about 18 mixed-core, while gemma-4-e2b is better mixed, 10.8 against
about 5.6.

**M23. Verify the core to capacity mapping per device. Never hardcode a mask.**
A Pixel 10a reports one prime core (`cpu7 cap=1024`); the Samsung reports two
(`cpu6-7 cap=1024`). Both used mask `80`, for different reasons.

## See also

`ANDROID_OPTIM_PROGRESS.md` carries a longer pitfalls list built from this campaign
specifically. `LEDGER.md` is the raw evidence trail.
