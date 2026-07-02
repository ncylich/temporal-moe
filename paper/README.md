# Paper — style & decisions (what Noah taught me)

The Temporal-MoE short paper (`main.tex` → `main.pdf`, author: Noah Cylich, noahcylich@gmail.com —
personal email on purpose, to avoid misattributing the work to an employer). This file records the
guidance Noah gave while we built it, so future edits keep the same quality bar. Rules first, each
with the reason it exists.

## Naming & precision
- **No internal codenames in prose.** Write "coarse-expert model (6 of 64 experts)" and
  "fine-grained model (18 of 192)", never "G1/G3". A codename is meaningless to a third-party reader.
- **Don't introduce notation the paper never uses.** We dropped "K = k"; just say `k`, the active
  experts. Extra symbols are cognitive load for nothing.
- **One data point is not a claim.** The 1e18 result where fine-grained temporal beat the
  fine-grained full MoE was a *single, unexplained* observation → at most one hedged clause
  ("sometimes the mechanism appears nearly lossless, outperforming its equally fine-grained MoE
  counterpart"), no proposed mechanism, no "robustness" framing.
- **Granularity framing (Noah):** the coarse model (6 of 64) is the head-to-head comparison with
  FLAME-MoE; the fine-grained model (18 of 192) carries the transition-to-target-sparsity story.
  Weight fine-grained slightly more in the narrative; keep actual-hardware specifics light.
- **Don't overstate; check the numbers before writing them.** Recovery is 72–82% (not a vague
  "65%"); at 1e18 temporal *recovers ~74%*, it does **not** "match" the full MoE. Re-verify claims
  when new controls/data land instead of jumping to a conclusion.

## Structure & prose
- **Consolidate.** Early-stage findings don't get their own section — the negative results were
  folded into "Outlook and next steps". Prefer fewer, denser sections.
- **Say each idea once.** No repeating the same fact/phrase across abstract, body, and caption.
- **Methods is general and split by concern:** one paragraph for the technique (the swap rule),
  a separate paragraph for memory/bandwidth. Don't cram it into one wall of text.
- **Justify the core mechanism in Methods:** the load-masking inequality (why one swapped expert
  hides behind the other `k-1`'s compute) belongs there; future ideas (router-early) get one line.
- **Add the theory caveat verbatim** where bandwidth numbers appear: "These calculations are
  theoretical. We need to also explore them in real kernels."
- **Few em dashes and colons.**
- **One-sentence nods** to related directions we're weighing (e.g. aux-free / DeepSeek-inspired
  selection biases) — mention, don't expand.
- **Keep the Vision section:** a short, less-academic closer (local-LLM dream, Kimi-K2.X-class
  locally on consumer GPUs) in Noah's own voice.

## Figures
- **On-page legibility is the priority, and you must inspect the *compiled PDF*, not the raw PNG.**
  Wide figures get downscaled into the column, so a 10pt matplotlib label renders at ~3pt. Fix with
  compact `figsize` + large fonts (~15pt in-figure ≈ 9–10pt on page = "small–medium" LaTeX).
- **Short titles + short axis labels in the figure; verbose detail goes in the LaTeX `\caption`.**
- **Captions live in LaTeX, not baked into the image.** Figures render caption-less via the
  `--no-caption` "paper mode".
- **Extend existing scripts with a non-default option** (e.g. `--no-caption`); don't fork scripts,
  hand-edit images, or hardcode.
- **Don't shrink the bibliography to save space** — a 4th page for references is fine. Keep font
  sizes consistent (the abstract was accidentally small once; match body size).
- **Raster (`expert_selection_per_token_*`) specifics:** circles not squares (overlap less); taller
  panels + *small, not tiny* dots so streaks stay distinct (dots ended at 60% of an earlier size,
  not smaller); label the top expert index (`0 / 31 / 63`), don't stop at 50; y-label "expert idx".
- **Regenerate from real data.** Noah pushes the data so graphs can be remade directly; when raw
  logs aren't local, redraw from the committed CSV stand-ins in `results/phase0/figure_data/`.
- **IsoFLOP standard encoding (use for every quality-vs-params graph):** color = method (dense gray,
  MoE blue, temporal green), shade = granularity (coarse normal / fine-grained dark), marker =
  compute budget (circle $10^{16}$, triangle $10^{17}$), equal line weight + opacity. One canonical
  figure/script: `fine_grained_vs_coarse_experts_isoflop*.png` from `plot_g3_curves.py`. Superseded
  coarse-only / dense-only isoFLOP figures and their scripts were deleted — don't reintroduce them.
- **Reuse that same hue+shade scheme on other quality graphs** (e.g. the 1e18 bars, `plot_1e18.py`):
  solid bars, no hatching (reads as noise); note any cross-data / different-split bars in the caption.

## Camera-ready TODO (deferred by Noah — not before final revisions)
- **Three seeds per 1e18 column** (all five Figure-1-right configs, including a LOCAL dense run):
  upgrades the error bars from two-seed ranges / method-matched estimates to real standard
  deviations, and the local dense run removes the last cross-data caveat. Deferred because the
  measured spreads are already tiny (temporal ±0.0024, MoE ±0.0064 nats) — the conclusion cannot
  flip; ~10-13 extra 1e18 runs (~2-3 days of H100) is only worth spending for camera-ready.

## Build
`cd paper && pdflatex main.tex` twice (uses `neurips_2024.sty` if present, else a plain-article
fallback). Figures are the `*_nocaption.png` variants from `scripts/phase0/plot_*.py --no-caption`.
