# Storage probes for the Pixel 10a temporal-MoE campaign

Cross-compile with the NDK, push to `/data/local/tmp/tmoe/`, run as shell (root only for
`pin.sh`/`unpin.sh`/`mon2.sh`):

    NDK=/opt/homebrew/share/android-ndk
    CC=$(ls $NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android31-clang | head -1)
    $CC -O2 -o probe probes/anat2.c        # bionic has pthreads in libc; do NOT pass -lpthread

| file | what it measures |
|---|---|
| `anat2.c` | **The one that matters.** Per-request cost ladder (size sweep, QD1, O_DIRECT) -> fits `latency = 163 us + 0.63 us/KiB`. Also device-cache check (there is none) and pure kernel path via page-cache-hot buffered reads. |
| `burst5.c` | Wall time to deliver ONE expert (648 KiB) under different decompositions, with optional memory load and optional shared DMA-destination buffers. Args: `<file> <nparts> <use_preadv> <nload> <separate_bufs> <base_off> <shared_dst>` |
| `probe.c` | Sustained throughput vs request size and queue depth. **Read the warning below before trusting it.** |
| `chk.c` | Checks whether a 648 KiB `preadv`/`pread` returns short (it does not; `max_sectors_kb=512` splits internally, not at the syscall boundary). |
| `pin.sh` / `unpin.sh` | Pin/restore `scaling_min_freq` to defeat DVFS. **Always restore.** |
| `mon2.sh` | Runs the engine with the UFS driver's `monitor` interface enabled, counting requests whose size == `monitor_chunk_size`. This is how the 512 KiB + 136 KiB split was proven. |

## Warnings earned the hard way

1. **Offsets must land in WRITTEN data.** The side-file is sparse; a read into a hole never
   reaches the device and returns zeros instantly (measured "3.4 GB/s"). `anat2.c` constrains
   offsets to `[300 MiB, size-slack)` for this reason.
2. **Check return values.** A probe that ignores a short read reports a smaller transfer as fast.
3. **Idle probes do not predict in-engine behaviour** -- the ranking of request shapes
   *inverts* under real compute load. Never size an engine change from these alone; use
   `LLAMA_TEMPORAL_FETCHPROF=1` inside the engine instead.

See `../../ANDROID_OPTIM_PROGRESS.md` § MEASUREMENT PITFALLS for the full rule list.
