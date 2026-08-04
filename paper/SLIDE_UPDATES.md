# Updating the deck against the current findings

Reference read: `tmoe-presentation.pdf`, 23 slides. Main deck 1 to 12, closing 13 to 14, appendix 15
to 20, templates 21 to 23.

Source of truth for every number below is
[`docs/research/mechanism/01-findings.md`](../docs/research/mechanism/01-findings.md), which was
rebuilt from the CSVs and corrected four claims in the process.

## A. Slides to update

### A1. Slide 9, "What do the experts become?" — retract the logit-lens claim

**This is the only slide carrying a claim we have since withdrawn**, so it should be fixed before the
deck is shown again.

The slide says full-MoE experts promote coherent word clusters while temporal experts have no
vocabulary preference, with median effective vocab 14,612 against 15,932 and "no word-list experts" in
the sharpest decile.

That result does not replicate. At 1e18 the temporal model writes *sharper* output distributions at
4 of 8 layers on the data-weighted metric and 0 of 8 on the static one, with no consistent direction
in the fine-grained pair. The cause was a capture defect: router logits were filed under a 1-based
layer number and expert output vectors under a 0-based module index, so every output vector was
attributed one layer too shallow. Recorded as defect A in `02-corrections.md`.

- **Cut** the logit-lens table and both lens bullets.
- **Keep** effective experts, pairwise cosine and pool coverage, all of which stand.
- **Replace the freed space** with the weight-distribution result, which is new and stronger: excess
  kurtosis 0.42 against 0.14 on the coarse 1e18 pair and 0.62 against 0.24 on the fine one, widening
  to 2.79 against 0.77 at the 99th percentile.

### A2. Slide 8, "How are experts chosen?" — swap in the newer figure

The per-expert scatter is fine but the arm-level figure is the stronger version of the same point and
did not exist when the deck was made.

- **Swap** `delexicalization_locus_scatter.png` for `arm_separation.png`. One point per model rather
  than per expert, 34 models, and it marks the empty separating band at **0.184 wide** on the token
  axis.
- **Add one line**: the separation is entirely on the token axis. Context AUC is 0.640 full MoE
  against 0.697 temporal, near enough that several temporal models sit below full-MoE ones. The
  constraint does not add context sensitivity, it removes the token signal. That framing is sharper
  than "inverted" and it is what the figure shows.
- **Check the quoted probe numbers.** The slide says 0.93 / 0.64 and 0.77 / 0.62. Per-arm medians at
  window w = k on document-disjoint splits are 0.884 / 0.640 and 0.585 / 0.697. The gap is probably
  the older position split, which leaked documents across the fit and score halves. Re-derive before
  reprinting.

### A3. Slide 10, "Is it stable?" — one number to re-check, one to add

- **Add**: nothing moves at 8 bits in either regime, and the temporal model degrades less from 16 to
  3 bits in all three matched pairs. The slide shows the curves but does not say the pairs are 3 for 3.
- **Re-check the gradient claim.** The slide says dense had the biggest spike. Our reading of
  `stability_gradnorms.csv` has `temporal_coarse_1e19` at a maximum of 12.47 against the full-MoE
  2.52, which is the largest in the file. One of the two readings is wrong, or they are different
  slices. Resolve before showing.

### A4. Slide 11, "What are the honest limits?" — extend the limits list

The dose curve and the +0.023 BPB ceiling are correct and unchanged. Add three limits the deck does
not currently admit:

- Every locus and lens measurement is one training seed per cell.
- Only one full-MoE run kept a router log, so every regime contrast in the serving section rests on a
  single baseline.
- Eight result files can never be regenerated, and two of the serving claims rest on them.

### A5. Slide 18, appendix, "Do you have to train from scratch?" — label the arms exactly

The bar chart is right and the headline holds. One check: the slide's 93.2% is labelled
"router + norms + LoRA r32", but the bake-off records router+LoRA at 91.4% and a separate
cross-entropy-surface arm at 93.2%. Confirm which arm the bar is before the number is quoted again.

## B. Slides to add

Ordered by how much they change what a listener concludes.

### B1. "Which layers should you free?" — the strongest new result

Currently no slide, though `slideA3_olmoe_ladder.png` exists unused.

The story is a controlled pair. On a 16-layer adapted model, layers 2 and 15 tie almost exactly on
single-layer damage, 0.1408 against 0.1408. Freeing them alongside layers 0 and 1 at identical memory
is not equivalent:

| free set | memory | BPB | downstream |
|---|---|---|---|
| {0,1} | +87.5% | 0.814440 | 0.5937 |
| {0,1,2} | +131.2% | 0.808615 | 0.5937 |
| {0,1,15} | +131.2% | 0.797810 | 0.6030 |
| {0,1,14,15} | +175.0% | 0.786275 | 0.6037 |

The training-free profile predicted layer 2 at 5.8 times layer 15. Trained, layer 15 wins by 0.0108
BPB and takes the better downstream score. Two further cells contradict the profile the same way.

**Do not choose free sets from single-layer damage.** This is a result about method, not just about
this model, and it is the most transferable thing in the recent work.

### B2. "What adaptation actually needs" — the failed ablations

Slide 18 shows four bars and reads as a ladder anyone would have guessed. The nulls are the finding:

| what is trained | recovery | |
|---|---|---|
| router only | 0.707 | floor |
| router + annealing R 64 to 8 | 0.708 | null |
| router + self-distillation from the free-routing teacher | 0.702 | null |
| router + RMSNorm gains | 0.914 | works |
| router + LoRA r=32 | 0.914 | works |
| router + LoRA + zone-confined anneal | 0.914 | null |
| full fine-tune | 0.934 | ceiling |

That is the bake-off. The embedding program ran six more, all null against the noise scale of their
own comparison:

| technique | result |
|---|---|
| per-layer embeddings, mechanism claim | **refuted**: locus moves 0.0031 against a spread of 0.0031, in the wrong direction |
| per-layer embeddings stacked on LoRA | 0.49σ, wrong side, at both ranks |
| calibrated initialisation, three variants | tie, and the strongest is **1.23σ worse** |
| sequential against joint training | 0.47σ |
| LoRA rank above 128 | not binding; r = 32 is 1.31σ worse |

Two lines worth saying aloud. **The jump is capacity, not schedule**: everything router-only lands at
0.70, everything with a degree of freedom outside the router lands at 0.91, nothing in between. And
**self-distillation fails for a reason you can state**: the teacher's free routing is exactly what the
constraint removes, so there is no signal to transfer.

Eight nulls is too many for one slide. Suggested split: the four-bar recovery chart plus the capacity
line on the main slide, and the null table in the appendix. The audience needs to know the recipe
survived a search, not what every arm of the search was.

**One trap to avoid on this slide.** The published locus figure for the cross-entropy surface is
0.0493 and the embedding cells read 0.093 to 0.096. Those come from different probes. Pairing them
shows an embedding effect that does not exist, and the findings document made exactly that error until
today. Measured with one probe: −0.0041 imposed, +0.0932 adapted, +0.0964 with embeddings.

### B3. "A perfect cache oracle is worse than a good one"

Replacing the demand estimate with perfect next-token foresight helps at coarse granularity, +6.6 to
+10.1 points over six runs, and hurts at fine, −2.6 to −11.2 over fourteen. Splits 20 of 20 by
granularity.

Counter-intuitive, memorable, and it redirects effort: the reading is that chasing instantaneous
demand destroys accumulated locality. Two caveats belong on the slide, not in the notes: granularity
is confounded with expert count, and none of the twenty runs kept a router log, so it cannot be
re-measured.

### B4. "Where the serving headroom actually is"

Sharpens what slide 5 asserts. Eviction policy is worth **6.5 to 9.7 points**, the whole gap from the
shipped rule to the offline optimum. Smoothing the demand estimate is worth **2.8x**. Figure exists:
`eviction_policy_headroom_belady_bound.png`, rebuilt so its labels no longer overlap.

### B5. "What the cost actually tracks" — optional, for a technical audience

Per-layer constraint cost against every routing profile, thirteen layers of one model: churn −0.91,
demand forecastability +0.78, hit rate +0.75, **contextual share +0.19**.

The lexical story does not explain the cost profile. Demand stability does, and the first three are
one factor measured three ways, intercorrelating 0.87 to 0.97. Worth a slide only if the audience
cares about mechanism over deployment.

## C. Not proposed, and why

- **Serving context sweep and bandwidth timeline**, slides 5 and 17. Systems results, unaffected by
  any of this work.
- **Slide 20, the alignment program.** Still accurate. Note that the mechanism findings document does
  not cover it at all, which is a gap in the document rather than in the deck.
- **Slide 19, mechanism extras.** The unmask/impose asymmetry it reports is now measured much more
  sharply, +0.4314 BPB imposed against +0.2006 unmasked at 1e19, but the slide is appendix and the
  headline slide already carries the point.

## Design questions before drafting

1. **Venue.** The deck closes on "Black Box → ICLR Main Paper?" and "Feedback?", so it reads as an
   advisor or lab talk rather than a conference version. Is the update for the same audience? That
   decides whether B5 belongs and how much of the appendix survives.
2. **Budget.** Main deck is 12 slides. Do the four or five additions replace existing slides, or does
   the deck grow to 16 or 17?
3. **Where does the layer-freeing work sit?** It is the strongest new result but it is about adapting
   a pretrained model, not about the from-scratch story the deck tells. Main line, or appendix
   promoted to the top of the appendix?
4. **How visible should the retraction be?** The logit-lens claim can disappear quietly, or the slide
   can say it did not replicate and why. The second is more honest and costs half a slide; it also
   invites a question about what else did not replicate, which has a good answer.
5. **Downstream evaluations.** They exist per task, ten tasks, and lambada goes to 0.000 under an
   untrained imposed constraint. That is the most visceral number in the entire program. Worth its
   own slide, or a line on B2?
