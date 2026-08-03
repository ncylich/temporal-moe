# Corrections to the published de-lexicalization write-up

**What changed, why, and what the number is now.** This is the delta between
[`archive/delexicalization-original.md`](archive/delexicalization-original.md): the write-up as it
stood before this program, and what the data supports now. Open this when revising the paper.

Each row names the defect that caused the change, because several of them share a cause and fixing
one without the others would leave the rest.

## 1. Claims that changed

| published claim | status | now | cause |
|---|---|---|---|
| Output lens: constrained experts are "barely distinguishable from no signal", unconstrained ones promote narrow vocabularies | retracted | The output-side regime difference does not replicate. At 1e18 the constrained model writes *sharper* distributions at 4 of 8 layers in the coarse pair on the data-weighted metric and 0 of 8 on the static one; the fine pair has no consistent direction either way | Defect A, plus §2 below |
| "Input side, output side, and structure now agree" | withdrawn | Input side and structure agree. The output side does not participate | follows from the row above |
| Selectivity and generalist fraction: 13% vs 54% generalists, stated as a uniform property | corrected | Pooling hid a regime change with depth. The two regimes are indistinguishable through layer 4; the unconstrained arm reaches 0% generalists only from layer 6 down | Defect B |
| Locus table row labelled "baseline (192E, w=18)" | corrected | That cell was only ever measured at w=32. Its sigmoid sibling has w=k and sits ~0.02 lower, which brackets the missing measurement | Defect C |
| Chance floors "0.500 ± 0.002 under permutation and circular shift" | half withdrawn | The iid permutation floor stands: worst deviation 0.0030 over 1,162 fits. The circular-shift floor is invalid and runs to +0.017, scaling with window width | Defect D |
| Held-out AUC from a 70/30 split | qualified | The split cut the flattened stream at a sequence *position*, so every document appeared in both halves. Re-measured on disjoint documents; the regime gap is an order of magnitude larger than the correction | Defect D |
| The 1e19 models are 9 layers deep | wrong | They are 14. Every "full depth" statement made before this was measured over layers 2 to 6 of 14 | Defect E |
| Per-layer replay numbers (E1 to E8) | replaced, not reproduced | The five runs they were computed on are absent from `MANIFEST.csv` and from disk. Current numbers cover a different population of 22 preserved logs | Defect F |

## 2. The defects behind them

**A: the capture attributed expert outputs one layer too shallow.** Router logits were filed under a
1-based layer number and expert output vectors under a 0-based module index, so the outputs stored at
key *j* belonged to layer *j+1* and the deepest MoE layer got none at all. Present in every capture
the script ever produced. **Only the output-lens family is affected**, logits and masks were always
keyed consistently, so locus, floors, structural, demand, oracle, frequency-stratification and
transfer are untouched. The capture now refuses to write a file whose two key spaces disagree.

**B, pooling across layers.** Several statistics were computed per (layer, expert) and then reported
as a single median over the whole stack. Where the regime difference varies with depth, the pooled
number is not wrong about the network as a whole but is wrong about where the effect lives.

**C: an unlabelled window variant.** The window sweep wrote three variants under names that decode
differently in two files (`base` means w=32 in one and w=k in the other). One cell was only ever run
at w=32 and was reported as if at w=k.

**D: the null control and the split shared a root cause.** Both were applied to the flattened
`[S·B]` stream, whose adjacent entries are adjacent *batch elements* rather than adjacent tokens. The
circular shift therefore never shifted along the token axis, and the 70/30 cut was a split on
sequence position. Six nulls were built to diagnose it; what survives is document-level association: a context feature is a moving average within one document and so a good document descriptor, which is
why the inflation scales with window width. Splits are now document-disjoint by default.

**E: an assumed depth.** Both plan documents assumed the 1e19 models were 9 layers. Layer lists are
now read from the artifact and nothing is hardcoded.

**F, preservation, not code.** No re-run can recover these. Four of the five runs behind the
published 1e16/1e17 locus rows and all five behind the replay numbers are absent from `MANIFEST.csv`
and from disk. `g3_moe_s0_1e16` is the one exception and has been re-captured.

## 3. What reproduced exactly

Worth recording, because the list of corrections above reads worse than the outcome:

- Every depth slope quoted in the original, to four decimal places.
- Tokens per expert per batch, identical across regimes: the optimization control in Appendix A.
- The pooled demand-forecastability values, which sit inside the per-layer ranges that replaced them.
- The dose curve endpoints.
- The per-layer cache hit-rate *shape*, within 0.04 on a different run in the same cell.

The corrections concentrate in one family, anything reading expert *outputs*, plus two
methodological controls that were measured on the wrong axis. The input-side result that carries the
paper is unaffected by all of them, and has since been extended from 8 arms to 34 and confirmed
causally.


## 3a. A verdict that moved twice

### The exempt-the-endpoints recommendation, **stands, on architectural grounds**

`archive/LAYER_LEXICALITY_ROUND2.md` §6 concluded: *"**The recommendation survives**: the first and
last MoE layers are by some margin the most expensive to constrain, so exempting them is the right
engineering call, and it is cheaper than any schedule we proposed."*

**That conclusion holds.** An intermediate version of this section said the training sweep overturned it; that reading
was wrong and is withdrawn. Three points, because this verdict has now moved twice:

1. **The schedule contrasts are underpowered, not negative.** Both endpoint-exemption schedules beat
   a uniform schedule at matched memory, exempt-last by −0.0080 CE, exempt-first by −0.0096, at
   roughly 0.7 se. The direction favours the recommendation in both arms; the magnitude is not
   established. "Not distinguishable from zero" is the correct reading of the statistic and the wrong
   reading of the evidence.
2. **The ordering supports it.** Among the single-layer arms the middle layer is cheapest to
   constrain and both endpoints cost more, in all three seeds; endpoints against middle is +0.0134 CE
   at 2.80 se.
3. **PLE supports it on a model deep enough to test it.** Freeing the first two MoE layers of an
   adapted 16-layer OLMoE buys 0.0003 BPB; adding the *last* layer to the free set buys 0.0169
   (`ple_ladder.csv`).

**What is refuted is the reason, not the recommendation.** The endpoint layers are not the
token-driven ones: the last MoE layer is the *least* lexical in the stack, and a magnitude-matched
perturbation carrying no lexical information reproduces 58 to 85% of the endpoint cost. So exempt the
endpoints for architectural reasons, and do not claim lexicality as the justification.

Recorded at this length because a fresh-context audit found the recommendation had been dropped
entirely from the reorganised set, and the entry written to restore it then overturned it on a
misread contrast.


## 4. Claims this program made and then withdrew

Listed here so the retraction is as visible as the claim was:

- **"The per-layer cost profile falls with depth over the interior"** (ρ = −0.886, p = 0.035). It flips
  sign on a seed replicate at the same budget and reverses at the next budget up. The measurement
  stands; the generalisation does not.
- **"Constraining the interior layer alone improves the model"** (−0.0120 CE at one seed). Across three
  seeds it changed sign to +0.0026 and is null.

Both were single-seed results reported before replication, which is the same failure mode as several
rows in §1.

Three more were asserted in `01-findings.md` during the reorganisation and corrected within days. They
are listed because they were committed claims, not drafts, and because the third is the most
instructive error in the program:

- **"The three statistics do not overlap between regimes."** Two of the three separate cleanly; context
  AUC overlaps across 0.633 to 0.679. The context probe finds real signal in both regimes, so its level
  was never the discriminator.
- **"There is no per-layer structure worth exploiting."** Written without reading `ple_ladder.csv`,
  which sits in the same directory as every other file the section cites and shows the opposite.
- **"The endpoint effect does not survive co-adaptation"**, supported by a *first versus last* contrast
  of +0.0014 at 0.2 se. A U-shape predicts first ≈ last: that contrast is the one the hypothesis says
  should be null, so its nullity confirms the shape's symmetry rather than refuting its existence. The
  contrast that tests a U, endpoints against middle, is +0.0134 at 2.80 se, in the predicted
  direction and consistent across all three seeds.

All three were found by recomputing from the CSVs rather than by rereading the prose, which is the
only method that has ever caught anything in this document set.
