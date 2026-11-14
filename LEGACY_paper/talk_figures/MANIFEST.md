# Talk figure package — MANIFEST

Self-contained assets for the 10-minute Temporal-MoE pitch. Slide content and all
spoken/reserve text live in `paper/SLIDES.md.md` (source of truth). Files are slide-prefixed.
"temporal" = the rolling-residency MoE (top-k resident, swap 1 expert per token). BPB = test-set
bits-per-byte (lower is better). Color standard: dense grey, baseline MoE blue, temporal green;
darker shade = fine granularity.

## Main slides
- **Slide 1** — no figure (title + one-line claim).
- **Slide 2** — `slide02_total_vs_active.png`. Log-y grouped bars, total vs active params for
  Qwen3-30B-A3B (30B/3B, ~10%) and Kimi K3 (2.8T/50B, ~1.8%). New matplotlib figure.
- **Slide 3** — no figure. Table built in the deck (prior-work reported numbers, see SLIDES.md.md).
- **Slide 4** — `slide04_swap_diagram.svg` (+ `.png` preview). Router evict/admit walkthrough,
  extracted from the one-pager poster. Standalone (xmlns + inline styles).
- **Slide 5** — `slide05_bandwidth_timeline.svg` (+ `.png` preview). Swap-cost timeline: k-1 RAM
  loads hide the one storage stream. Extracted from the poster, standalone.
- **Slide 6** — `slide06_isoflop_highlight.png`. 2x2 isoFLOP panels with temporal-fine (dark green),
  coarse MoE (light blue), dense (grey) emphasized; temporal-coarse + fine-MoE faded (still visible).
- **Slide 7** — `slide07_serving_context_sweep.png` (right half). LEFT half is a serving-rows table
  built in the deck (see SLIDES.md.md).
- **Slide 8** — `slide08_delex_locus.png`. Per-expert token-probe vs context-probe AUROC scatter.
- **Slide 9** — `slide09_expert_streaks.png`. Experts-active-per-token raster (streaks). The
  logit-lens table on this slide is typeset in the deck. Backup: `backup_locality_overlap.png`.
- **Slide 10** — `slide10_fakequant.png`. CE cost of quantizing routed experts vs bit width.
- **Slide 11** — `slide11_dose_curve.png`. Held-out BPB vs resident experts R (memory-quality frontier).
- **Slide 12** — `slide12_blackbox_joke.png` (+ `.svg`). Deadpan black-box -> "???" -> ICLR paper.
- **Slide 13** — no figure (the three closing questions).

## Appendix slides
- **A1** — no figure. Mac (M4 Pro) results table, built in the deck (see SLIDES.md.md).
- **A2** — `slideA2_serving_context_sweep.png` + `slideA2_vanilla_floor.png`. The prefill-row table is
  typeset in the deck.
- **A3** — `slideA3_olmoe_ladder.png`. OLMoE recovery ladder (70.7 / 91.4 / 93.2 / 93.4%), 93.2%
  (+LoRA r32) emphasized as deployable; subtitle notes cold impose = +2.08 BPB. No dense OLMo-1B bar.
  New matplotlib chart (supersedes the text-fallback ladder in SLIDES.md.md).
- **A4** — `slideA4_expert_streaks.png` + `slideA4_locality_overlap.png`.
- **A5** — no figure. Negative-results loss-attempt table, built in the deck (see SLIDES.md.md).

## Provenance / regeneration
- Slides 8–11, 7, A2, A4 PNGs are copied verbatim from `paper/figures/*_nocaption.png`.
- Slide 6 via `python3 analysis/plots/plot_isoflop_panels.py --highlight-deck --no-caption` (new
  `--highlight-deck` flag; default output verified md5-identical to the committed paper figure).
- Slides 2, A3 via `python3 analysis/plots/plot_talk_extras.py` (new talk-only script).
- Slides 4, 5 extracted from `paper/Temporal MoE Research Poster.zip` -> `Temporal-MoE-OnePager.html`.
- Slide 12 SVG hand-authored; PNG via qlmanage.

## Tables to typeset in the deck (figure IS a table — build from SLIDES.md.md)
Slide 3 (prior work), Slide 7 (serving-rows, left half), Slide 9 (logit-lens 16k-vocab),
A1 (Mac results), A2 (prefill row), A3 (recovery-ladder numbers — a chart also ships), A5 (loss attempts).

## Caveats
- `slide04_*.png` / `slide05_*.png` are qlmanage previews and carry square whitespace padding around
  the wide/short diagrams. The **.svg is the true deliverable** (vector, crops cleanly); PNGs were
  inspected and confirmed complete (shapes + text, not blank). Fonts fall back to sans-serif (poster's
  IBM Plex Sans is not embedded); layout unaffected.
- No fallbacks were needed for extraction; `poster/scraps/tl3.png` (timeline crop) went unused.
- Nothing was hand-edited as an image. No blocked variants.
