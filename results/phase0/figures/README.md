# Figures

Result figures for the temporal-MoE study. **Every figure here must be understandable on its own** —
by someone who has never seen this project or any discussion of it.

## Conventions (follow these when adding or regenerating a figure)

- **File names:** descriptive and context-independent — name by *what the graph shows*, not by
  internal shorthand. No `A/B/C` letters, no `probe_*`, no run codenames.
- **No internal jargon in titles / legends / annotations.** Translate to human terms:
  - **Shape codes → active non-embedding params.** sm1=0.77M, s0=1.36M, s1=3.81M, s2=8.12M,
    s3=14.77M, s4=24.29M (fine-grained variants: sm1=0.81, s0=1.42, s1=3.91, s2=8.23, s3=15.09M).
    Write "8.1M active", never "s2".
  - **G1 / G3 → "coarse experts (6 of 64)" / "fine-grained experts (18 of 192)"** (fine-graining
    splits each expert 3×: 64→192 experts, top-6→18).
  - **Spell out methods:** "temporal" = *rolling residency* (keep the top-k experts resident, swap 1
    in per token); "full MoE" (all experts available); "dense baseline" (no experts).
  - **Budgets:** write 1e16/1e17/1e18 as compute budgets ("at 10^17 FLOPs"), not bare numbers.
  - No `A —` / `B —` style prefixes — use a descriptive title.
- **Caption on every figure:** a 1–2 sentence caption at the bottom (`fig.text(0.5, 0.01, ...)`) stating
  what's plotted, what the axes mean, and **which direction is better**.
- **Metric:** plot **bits-per-byte (BPB, lower better)** for our 16k-tokenizer sweeps; use **raw
  cross-entropy** only when matching a paper's exact tokenizer (the 1e18 / 50k-vocab replication).
- **Preferred IsoFLOP format:** one combined single axes with both budgets (dashed = 1e16, solid =
  1e17; color = method), not separate side-by-side panels.

Generators live in `scripts/phase0/plot_*.py`.
