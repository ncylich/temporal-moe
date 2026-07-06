# Why the Temporal Constraint Wins: De-Lexicalization of Routing

## 1. The puzzle and the setup

Rolling residency restricts each token to the currently resident expert set, removing routing
freedom, yet at 1e18 FLOPs the temporally constrained model achieves *lower* validation loss
than the unconstrained MoE baseline of identical architecture, data, and compute, at both
granularities (coarse: CE 3.9121 vs 3.9209, fine: 3.9768 vs 4.0087, both far below the dense
floor of 4.137, with a 1e19 replication in progress). A constraint that improves generalization
must be suppressing a harmful behavior, and this section identifies it.

We analyze the softmax-aux baseline and the temporal model at the two budgets where full
training artifacts exist: 192 experts with top-18 routing at 1e16, and 64 experts with top-6 at
1e17 (16k vocabulary). At these small budgets the baseline holds a loss edge (1.4519 vs 1.4750
test BPB at 1e16, 1.269 vs 1.282 at 1e17), so the analyses below characterize what the
constraint does to routing, while its quality benefit emerges at larger budgets (Section 5 shows
the constraint is nonetheless load-bearing even where it wins). One experiment additionally used
an unconstrained sigmoid-router control, which appears only in the table of the experiment that
used it. All statistics derive from three sources on a fixed evaluation batch of 64 sequences of
2048 tokens (N = 131k tokens): router probabilities, expert weight matrices, and expert output
vectors. Every table carries its own calibration, either a measured chance floor or the
within-regime seed spread (about 0.01 in the selectivity statistic and two points in generalist
fraction, from independent-seed replicates of both regimes, and the unconstrained pattern is
recipe-robust: sigmoid and aux-free router variants match the baseline on every metric below).

## 2. The constraint reshapes routing, not experts

We first ask whether the constraint changes what experts *are* (their weights) or how they are
*used* (their traffic). Let $g_e(t)$ be the router softmax probability of expert $e$ at token
$t$. Three measurements:

**Expert selectivity.** Renormalize each expert's gate mass into a distribution over token
positions, $q_e(t) = g_e(t) / \sum_{t'} g_e(t')$, the fraction of expert $e$'s total usage
contributed by token $t$. Selectivity is the normalized inverse Simpson concentration
$$\mathrm{PR}_e = \frac{1}{N \sum_t q_e(t)^2} \in (0, 1],$$
the fraction of the stream effectively feeding expert $e$: uniform usage gives 1, and usage
concentrated on $m \ll N$ positions gives about $m/N$. We report the median over experts and the
generalist fraction $|\{e : \mathrm{PR}_e > 0.5\}| / E$.

**Router flatness.** Per-token routing entropy normalized to $[0,1]$:
$\bar{H} = \mathbb{E}_t[-\sum_e g_e(t) \ln g_e(t)] / \ln E$.

**Weight identity.** Flatten each expert's FFN weights into a vector $w_e$, form the centroid
$\bar{w}$, and report each expert's cosine distance to it, $d_e = 1 - \cos(w_e, \bar{w})$, and
the median pairwise $\cos(w_e, w_{e'})$.

| model (scale) | median PR | generalist % | $\bar{H}$ | $d_e$ (mean) | pairwise cos |
|---|---|---|---|---|---|
| baseline (192E) | 0.30 | 2 | 0.88 | 0.90 | 0.005 |
| temporal (192E) | 0.66 | 65 | 0.95 | 0.88 | 0.010 |
| baseline (64E) | 0.34 | 13 | 0.86 | 0.84 | 0.009 |
| temporal (64E) | 0.52 | 54 | 0.93 | 0.84 | 0.012 |

Baseline experts each draw usage from a narrow recurring slice of the stream (2% generalists at
192E), temporal experts from most of it (65%), under visibly flatter routing, and the regime gap
is ten to forty times the seed spread on every routing metric. The weight geometry, by contrast,
is indistinguishable across regimes and seeds: experts remain equally distinct and near
orthogonal either way. The constraint acts on traffic, not on expert identity, which sharpens
the question: if a temporal expert is not specialized on a token slice, what is it specialized
on?

## 3. The locus of specialization moves from token to context

We answer by testing what *predicts* an expert's firing. Define $y_e(t) = 1$ if expert $e$ is
among the top-k assigned to token $t$. Build two feature sets from the model's own input
embeddings: the current token alone, $x_{\mathrm{tok}}(t) = E[x_t]$, and the surrounding context
with the current token excluded,
$x_{\mathrm{ctx}}(t) = \mathrm{mean}\{E[x_{t'}] : 0 < |t'-t| \le w\}$. The exclusion is the
essential control, preventing context features from encoding the token itself. The window is one
residency lifetime, $w = k$: under one swap per token an expert admitted to the cache survives
about $k$ tokens, so this is precisely the context the resident set can exploit. For each expert
we fit two logistic classifiers, $y_e \sim x_{\mathrm{tok}}$ and $y_e \sim x_{\mathrm{ctx}}$, on
the first 70% of tokens and report held-out AUC on the last 30%. Each row carries its measured
chance floor: the same classifiers refit on null labels (iid permutations and circular shifts of
at least 1000 tokens, the latter preserving the labels' residency-induced autocorrelation) give
$0.500 \pm 0.002$ everywhere, so every entry is real signal.

| model | median $A_{\mathrm{tok}}$ | median $A_{\mathrm{ctx}}$ | chance floor (tok / ctx) | context dominated |
|---|---|---|---|---|
| baseline (192E, $w{=}18$) | 0.93 | 0.64 | 0.501 / 0.499 | 1% |
| temporal (192E, $w{=}18$) | 0.62 | 0.77 | 0.500 / 0.498 | 91% |
| baseline (64E, $w{=}6$) | 0.84 | 0.59 | 0.499 / 0.502 | 1% |
| temporal (64E, $w{=}6$) | 0.60 | 0.68 | 0.501 / 0.501 | 85% |

The baseline router implements a near deterministic token-to-expert lookup: the current token
predicts expert firing at AUC 0.84 to 0.93 (+0.34 to +0.43 above floor) and context never
overtakes it for more than 1% of experts. The temporal model cannot implement that lookup,
because a token must be served by whichever experts are resident, and its token AUC falls to
0.60 to 0.62. Against the floor this is a reduction, not an erasure: temporal experts retain a
weak but genuine lexical signal (+0.10 to +0.12, roughly a quarter of the baseline's), while
their contextual signal (+0.26 at 192E) dominates it, with 85 to 91% of experts better predicted
by their surroundings than by the token they process and the effect growing monotonically with
depth. We call this *de-lexicalization*. The result is robust to the window: across
$w \in \{\lfloor k/2 \rfloor, k, 32\}$ the fine pair is flat (88 to 94% context dominated) and
the coarse pair is context dominant at both residency-scale windows (85 to 86%, context AUC
peaking at $w = k$), washing out to balanced only at $w = 32$, five times its cache lifetime.
Context is the transferable feature, autocorrelated within documents and shared across surface
forms, and we identify this as the regularization behind the loss improvement at scale. It
equally explains why temporal routing demand is far more forecastable from history (AUC 0.85 vs
0.64 in our demand prediction analysis): context persists across neighboring tokens, token
identity does not.

## 4. The output lens agrees

The two analyses above examine what makes an expert fire. As an independent check we examine
what an expert *writes*. For each expert, average its output vectors over the tokens actually
routed to it, project through the final norm and unembedding $U$, and form
$p_e = \mathrm{softmax}(U \cdot \mathrm{norm}(v_e))$, the vocabulary distribution the expert
promotes on its real traffic. Sharpness is the effective vocabulary
$V_{\mathrm{eff}}(e) = \exp H(p_e)$, from 1 (a single word) to the vocabulary size (16k here).
The no-signal reference comes from the method's own failure mode: projecting raw weight columns
with uniform weights, without conditioning on data, reads about 15,990 for every expert in both
models, because averaging unactivated columns cancels their directions and mid-network
projections are rotated relative to the output basis. All comparisons are therefore data
weighted and within layer.

| model (192E) | mean $V_{\mathrm{eff}}$ | sharpest decile | no-signal reference |
|---|---|---|---|
| baseline | 14,612 | 9,683 | 15,990 |
| temporal | 15,932 | 15,342 | 15,990 |

Baseline experts promote markedly narrower vocabularies (about 1,400 effective words of lexical
preference at the mean and over 6,000 in the sharpest decile, whose top promoted tokens read as
coherent lexical clusters). Temporal experts are barely distinguishable from no signal (about 60
at the mean) and contain no word-list experts even in their extreme tail. One expert's write is
a nudge rather than a full prediction, so the mean shifts are modest, but input side, output
side, and structure now agree.

## 5. The mechanism is load-bearing and dose-tunable

If de-lexicalization were a pure training-time regularizer, the mask should be removable at
inference. A constraint swap tests this: evaluate every trained model under the *other* regime
(unmasking a temporal model means making every expert always resident, and the converse imposes
rolling residency on an unconstrained checkpoint), with each pair's native evaluation protocol.

| trained model | native loss | cross regime | delta |
|---|---|---|---|
| temporal, 192E at 1e16 | 1.4750 (masked) | 1.5744 (unmasked) | +0.10 |
| temporal, 64E at 1e17 | 1.2821 (masked) | 1.4063 (unmasked) | +0.12 |
| temporal, 1e18 (the winning case) | 3.9037 (masked) | 4.3890 (unmasked) | +0.49 |
| unconstrained control (sigmoid, 192E at 1e16) | 1.4499 | 1.6902 (imposed) | +0.24 |
| baseline (64E at 1e17) | 1.2690 | 1.8789 (imposed) | +0.61 |

Both directions hurt, so the advantage is serving co-adaptation, not better weights: even at
1e18, where the temporal model wins, unmasking collapses it below both its masked self and the
unconstrained baseline. The asymmetry makes the mechanism causal: imposing residency on lexical
routers costs two to five times more than unmasking contextual ones, exactly as Section 3
predicts, since a token denied its bespoke expert has nowhere to go, while a contextual expert
serves its neighborhood regardless.

The constraint also admits a dose. Decouple the cache size $R$ from the active top-k: the cache
holds $R$ experts (cold filled with the top $R$, evolving under the same one-swap-per-token
dynamics) and the router selects its top-k among residents, so $R = k$ is the maximal constraint
studied above and $R = E$ recovers the baseline recipe exactly, with FLOPs identical at every
$R$. Training from scratch at 1e16 (192 experts, k = 18, the baseline's own softmax-aux router
throughout):

| $R$ | 18 (= k) | 36 | 72 | 128 | 192 (= baseline) |
|---|---|---|---|---|---|
| test BPB | 1.4750 | 1.4736 | 1.4681 | 1.4580 | 1.4519 |
| effective experts | 183.9 | 183.4 | 181.8 | 186.4 | n/a |

Loss falls monotonically toward the baseline as the constraint loosens, confirming that at this
budget the constraint is a pure quality cost whose regularization pays only at larger scale, and
expert diversity is preserved at every dose, so the constraint acts on usage throughout. Because
$R$ is the number of experts held in fast memory and compute is fixed, the curve is also the
serving memory-quality frontier: the maximal constraint costs +0.023 BPB versus the baseline at
roughly one tenth of the routed-expert memory, and about three quarters of that gap is recovered
by $R = 128$, two thirds of the memory.

## 6. Summary

Baseline MoE routers learn a lexical shortcut, binding experts to token identities (input side:
the token predicts firing at AUC 0.93, output side: sharp promoted-word clusters). The shortcut
is easy to learn but does not generalize. Rolling residency makes it unrepresentable, a token is
served by whoever is resident, without touching what experts can compute (weight geometry
unchanged). Forced off the shortcut, experts reorganize around context, the feature that
transfers, which simultaneously explains the loss advantage at scale, the temporal coherence and
forecastability of routing demand, and why cacheability exists at all. The mechanism is
intrinsic to the trained model rather than a removable regularizer, and its strength is a single
dial that trades quality against serving memory along a clean monotone frontier.

## Appendix A: an optimization control

A mundane alternative explanation is gradient noise: if temporal sequences use fewer distinct
experts, each expert might receive more tokens per update, giving lower gradient variance
without any representational change. The temporal model does use fewer distinct experts per
sequence (158 vs 192 at the fine scale), but the per-expert update size is fixed by the
architecture: top-k routing assigns exactly $k/E$ of each batch's tokens to each expert on
average, however those assignments are distributed in time. Measured tokens per expert per batch
are identical across regimes (12,288 fine, 3,072 coarse). The hypothesis is rejected.

## Appendix B: baseline provenance

The softmax-aux baseline used throughout is a seed-consistent retrain (seed 1234, matching every
other cell) with a preserved routing trace. The originally recorded baseline run (test BPB
1.4585, single seed, artifacts not preserved) serves as a cross-machine anchor: the retrain's
1.4519 sits within the family's seed spread of it. Its sigmoid-router and aux-free variants,
measured for other experiments, match the baseline on every interpretability metric (selectivity
median 0.25 to 0.28, token AUC 0.94, lens sharpest decile 13,431), so the lexical regime is a
property of unconstrained routing generally rather than of one recipe.
