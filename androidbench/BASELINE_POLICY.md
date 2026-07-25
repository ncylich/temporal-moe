# Baseline policy for on-device temporal-MoE benchmarks

> Companion: **`../ANDROID_OPTIM_PROGRESS.md` § MEASUREMENT PITFALLS** lists the 12
> benchmarking mistakes actually made in this campaign, as rules. Read it first.

**Standing rule. Applies to every number we publish from `androidbench/`.**

## The rule

The baseline is **the largest total expert count E that fits fully resident on the device,
at the same active expert count K and the same per-expert width**.

If the full model fits on the device, the baseline is the full model, fully resident.
If it does not fit, we do **not** shrink to a model that trivially fits — we use as many
total experts as the device can actually hold, and we say what that number is.

On the Pixel 10a (7.75 GB RAM, Q4_0, K=18, d_ff=384, 45 layers, n_embd=1024) that is
**E=112** (~3.5 GB). The temporal model under test is **E=192** (~5.9 GB), which does not
fit resident — hence the streaming engine exists at all.

## Why: E = K is not an MoE

A model with `E == K` (e.g. the retired `e18` variant: 18 total experts, 18 active) is
**dense**, not sparse. Every expert fires on every token, all experts sit contiguously in
one small tensor, and the access pattern is perfectly predictable. It is the friendliest
possible memory case and flatters the baseline for a reason that has nothing to do with
the technique under test. Measuring against it overstates the cost of temporal streaming.

The `e18` variant and all of its artifacts have been **removed** from this repo and from
the device. Do not reintroduce it. (Historical numbers that used it are still in
`LEDGER.md` for provenance and are marked there.)

## Why a big pool is not itself a penalty — measured

Pool size is free. Fully resident, repacked, cool-gated, n=3, Pixel 10a at 1.95 GHz:

| config | decode tok/s | per-GEMV | expert tensor |
|---|---|---|---|
| E=18, K=18 (dense — retired) | 34.81 ± 0.96 | 6.79 µs | 3.9 MB |
| **E=112, K=18 (sparse baseline)** | **35.83 ± 0.93** | 6.88 µs | 24.2 MB |

A 6x larger expert pool with real sparse routing costs **nothing** (the two are within
each other's error bars). So using E=112 instead of E=18 does not handicap the baseline —
it just removes the dense-model artifact. There is no honest reason to prefer E=18.

## Consequences for the reported ratio

Ratios are quoted against the E=112 resident ceiling measured **in the same session, on
the same binary, at the same clocks**. Never against a ceiling from another day, another
model, or another kernel (see the repack lesson in `LEDGER.md` S3-19: comparing a repacked
baseline to a non-repacked temporal run inflated the apparent gap by ~1.33x).

## Checklist before publishing any baseline number

1. Same binary, same session, same cool-gate as the temporal arm.
2. Same kernel family (both repacked, or both not — never mixed).
3. `E` is the largest that fits resident; state it explicitly next to the number.
4. `K` and `d_ff` identical to the temporal arm, so per-expert arithmetic is identical.
5. `fetches=0` in the pool line, proving the baseline really is fully resident.
