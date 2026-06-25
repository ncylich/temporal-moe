# Findings: VRAM/RAM↔SSD cache-conditional offload is disqualified at batch-1

**Decision: do not implement [`PLAN.md`](./PLAN.md).** A bandwidth analysis (below) shows that
on this hardware — and on Apple Silicon — a single offloaded (SSD-resident) expert per layer
cannot be hidden behind the compute of the resident experts at batch size 1. Cache-conditional
routing only lowers the miss *rate*; it cannot reduce the per-miss stall, which here is several×
a whole layer's compute. So the method yields no usable speedup toward the resident ceiling in
this regime. The reusable artifact is this analysis, not an implementation.

## The governing inequality

At batch-1 decode the expert FFN is **bandwidth-bound** (each weight read once, multiplied by a
single token — no arithmetic intensity). With prefetch overlapping load and compute, per-layer
time = `max(T_compute_resident, T_load_offloaded)`. An offloaded miss is hidden only if:

```
n_ssd · S_e / BW_offload  ≤  n_resident · S_e / BW_fast
        ⟺   n_ssd / n_resident  ≤  BW_offload / BW_fast
        ⟺   experts of resident compute needed to hide ONE miss  =  BW_fast / BW_offload
```

`S_e` cancels (a miss is the same bytes from either tier). So the cost of one miss, measured in
"resident-expert compute-times," is just the **bandwidth ratio of the two tiers** — and the
budget available to pay it, per layer, is **at most the number of active experts, `k`**.

For Gemma 4 26B A4B: `k = 8` of `128` experts, `30` layers, one expert ≈ 5.9 MB int8.

## Per-platform numbers

| Platform (hierarchy) | BW_fast | BW_offload | Crossover miss\* = offload/fast | **Resident experts needed to hide 1 miss** = fast/offload | Available to hide behind (`k`) | Verdict |
|---|---|---|---|---|---|---|
| A6000 VRAM↔SSD | 768 GB/s | ~7 GB/s | ~0.9% | **~110** | 8 | impossible (14× short) |
| A6000 VRAM↔RAM (PCIe) | 768 GB/s | ~32 GB/s | ~4.2% | **~24** | 8 | impossible (3× short) |
| **M4 Pro 24 GB RAM↔SSD** (memory-bound) | 273 GB/s | ~5 GB/s | ~1.8% | **~55** | 8 | impossible (7× short) |
| M4 Pro, generous (compute-bound est.) | — | — | — | **~20** | 8 | impossible (2.5× short) |
| Paper mobile (Qwen, RAM↔Flash) | ~50 GB/s | ~4 GB/s | ~8% | ~12.5 | 4 | short, but won 2× **vs LRU** (not vs resident) |

\*Crossover miss rate = the per-token miss fraction below which loads fully hide. Even one miss
in a layer of `k=8` is a `1/8 = 12.5%` miss rate — already far above every crossover here.

**This is your conclusion, quantified:** the best "this-machine" case (A6000 + RAM offload)
needs **~24** resident experts of compute to hide one miss but offers **8**; the Mac needs
**~20–55** and offers **8**. Even generously counting attention + the dense MLP as extra
hideable work (~17 expert-equivalents of compute per layer), it is still short of the ~20–55
required. **A single SSD miss stalls the layer by ~3× (A6000+RAM) to ~7× (Mac, memory-bound);
layers that pull 2 new SSD experts double that.**

## Why so few experts makes it hopeless

The hideable budget per layer is capped at `k` because only the `k` selected experts run. You
cannot "add more resident compute" to hide a miss — the architecture activates 8 of 128, period.
So the only way to satisfy `n_ssd/n_resident ≤ BW_offload/BW_fast` is `n_ssd ≈ 0` — i.e., a
~0–1% miss rate. But at 50–75% cache with 128 experts and top-8, the realistic miss rate is the
paper's own ~7–21% — **5–20× over budget**, and with high variance: most painful exactly on the
layers that happen to need 2 fresh SSD experts.

## The counterintuitive part: faster hardware is worse

The per-miss cost is `BW_fast/BW_offload`. A bigger GPU has a *faster* fast tier (768 GB/s VRAM
vs the paper's 50 GB/s mobile RAM), so it races through resident experts and leaves the slow
storage *relatively* further behind. That is why the paper's modest Snapdragon (fast tier only
~12× its Flash) could show a 2× speedup while a 768 GB/s GPU (≥110× its SSD) cannot. **The
paper's 2× was relative to an LRU baseline, never relative to a resident model** — consistent
with the unmeasured fully-resident gap noted in
[`../docs/research/cache-conditional-experts.md`](../docs/research/cache-conditional-experts.md).
Our goal (useful speed toward resident) is exactly the comparison the method can't win here.

## The only escape: batch the tokens (Temporal MoE)

The inequality has one free lever we are not using at batch-1: **reuse a loaded expert across
`B` tokens.** Then a loaded expert serves `B` tokens of compute, multiplying the budget:

```
experts needed to hide 1 miss  →  (1/B) · BW_fast / BW_offload
        ⟺   crossover miss_rate*  ≈  B · BW_offload / BW_fast
```

At `B≈16` the Mac crossover rises from ~1.8% to ~29%, and the A6000+RAM case from ~4% to ~67% —
i.e., the miss cost finally drops below the available compute. **This is precisely the
[Temporal MoE](../docs/research/temporal-moe.md) thesis**, and this analysis is independent
confirmation that *windowed, batched* expert reuse — not per-token cache-conditional routing —
is the only thing that makes single-machine expert offload viable on high-bandwidth compute.

## Status

`PLAN.md` is shelved (not executed). No dependencies installed, no weights downloaded. The
takeaway for the research program: drop batch-1 cache-conditional offload; pursue the batched
Temporal-MoE direction, where the crossover scales with `B`.
