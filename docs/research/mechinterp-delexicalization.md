# Why the Temporal Constraint Wins: De-Lexicalization of Routing

Rolling residency restricts each token to the currently resident expert set, yet at 1e18 FLOPs
the temporally constrained model achieves *lower* validation loss than an unconstrained MoE of
identical architecture and compute, at both granularities (coarse: CE 3.9121 vs 3.9209, fine:
3.9768 vs 4.0087, both far below the dense floor of 4.137, 50k vocabulary; a 1e19 replication is
in progress). A constraint that improves
generalization must be suppressing a harmful behavior. This section identifies that behavior. We
analyze matched temporal and unconstrained pairs at the two smaller budgets where full training
artifacts are preserved: 192 experts top-18 at 1e16 and 64 experts top-6 at 1e17, both on the 16k
vocabulary, using a fixed evaluation batch of 64 sequences of 2048 tokens (N = 131k tokens). At
these small budgets the unconstrained model still holds a small loss edge (1.4499 vs 1.4750 test
BPB at 1e16, 1.269 vs 1.282 at 1e17), so the analyses below characterize what the constraint
does to routing, while the quality benefit of that change emerges at larger budgets. All
statistics are computed from three sources: the router probabilities on this batch, the expert
weight matrices, and the expert output vectors on this batch.

## Routing concentration and expert identity (P1)

We first ask whether the constraint changes what experts *are* (their weights) or how they are
*used* (their traffic). Let $g_e(t)$ denote the router softmax probability of expert $e$ at token
$t$. Three measurements:

**Expert selectivity.** Renormalize each expert's gate mass into a distribution over token
positions, $q_e(t) = g_e(t) / \sum_{t'} g_e(t')$, so that $q_e(t)$ is the fraction of expert
$e$'s total usage contributed by token $t$. Selectivity is the normalized inverse Simpson
concentration
$$\mathrm{PR}_e = \frac{1}{N \sum_t q_e(t)^2} \in (0, 1],$$
the fraction of the token stream that effectively feeds expert $e$: uniform usage gives
$\mathrm{PR}_e = 1$, and usage concentrated on $m \ll N$ positions gives
$\mathrm{PR}_e \approx m/N$. We report the median over experts and the generalist fraction
$|\{e : \mathrm{PR}_e > 0.5\}| / E$.

**Router flatness.** The per-token routing entropy, normalized to $[0,1]$:
$\bar{H} = \mathbb{E}_t\big[ -\sum_e g_e(t) \ln g_e(t) \big] / \ln E$.

**Weight identity.** Flatten each expert's FFN weight matrices into a vector $w_e$, form the
centroid $\bar{w} = \frac{1}{E}\sum_e w_e$, and compute each expert's cosine distance to the
centroid, $d_e = 1 - \cos(w_e, \bar{w})$, together with the pairwise similarity
$\cos(w_e, w_{e'})$ over all pairs. These measure whether experts drift toward a common average
function or remain distinct.

| model (scale) | median PR | generalist % | $\bar{H}$ | $d_e$ (mean) | pairwise cos (median) |
|---|---|---|---|---|---|
| temporal (192E, seed 1) | 0.66 | 65 | 0.95 | 0.88 | 0.010 |
| temporal (192E, seed 2) | 0.65 | 67 | 0.95 | 0.87 | 0.012 |
| unconstrained sigmoid (192E, seed 1) | 0.28 | 5 | 0.87 | 0.90 | 0.007 |
| unconstrained sigmoid (192E, seed 2) | 0.27 | 5 | 0.87 | 0.89 | 0.007 |
| unconstrained aux-loss (192E) | 0.25 | 12 | 0.85 | 0.88 | 0.010 |
| temporal (64E) | 0.52 | 54 | 0.93 | 0.84 | 0.012 |
| unconstrained (64E) | 0.34 | 13 | 0.86 | 0.84 | 0.009 |

The replicate rows put the noise band in the table itself: independent seeds agree to about 0.01
in PR, 2 points in generalist fraction, and 0.003 in routing entropy *within* each regime, while
the temporal-vs-unconstrained gap is ten to forty times larger on every routing metric. The
weight geometry is indistinguishable across regimes and seeds alike: experts remain equally
distinct and near orthogonal either way. What changes is traffic. Unconstrained experts each draw their usage
from a narrow recurring subset of the stream, while temporal experts draw from most of it, under
visibly flatter routing. The constraint therefore acts on routing statistics, not on expert
identity, which narrows the question: if a temporal expert is not specialized on a slice of
tokens, what is it specialized on?

## The locus of specialization (P2)

We answer by testing what *predicts* an expert's firing. For each expert define the binary label
$y_e(t) = 1$ if $e$ is among the top-k experts assigned to token $t$. Construct two feature sets
from the model's own input embeddings: the current token alone, $x_{\mathrm{tok}}(t) = E[x_t]$,
and the surrounding context with the current token excluded,
$x_{\mathrm{ctx}}(t) = \mathrm{mean}\{ E[x_{t'}] : 0 < |t' - t| \le w \}$. The exclusion is the
essential control, preventing context features from encoding the token itself. We set the window
to one full residency lifetime, $w = k$: under one swap per token an expert admitted to the cache
survives about $k$ tokens, so this is precisely the context the resident set can exploit. For
each expert we fit two logistic classifiers, $y_e \sim x_{\mathrm{tok}}$ and
$y_e \sim x_{\mathrm{ctx}}$, on the first 70% of tokens and report held-out AUC on the last 30%:
$A_{\mathrm{tok}}(e)$ and $A_{\mathrm{ctx}}(e)$. An expert bound to token identity yields high
$A_{\mathrm{tok}}$, an expert bound to context yields high $A_{\mathrm{ctx}}$.

| model | median $A_{\mathrm{tok}}$ | median $A_{\mathrm{ctx}}$ | chance floor (shuffled labels) | context dominated |
|---|---|---|---|---|
| unconstrained (192E, $w{=}18$) | 0.94 | 0.63 | 0.499 / 0.501 | 0% |
| temporal (192E, $w{=}18$) | 0.62 | 0.77 | 0.500 / 0.498 | 91% |
| unconstrained (64E, $w{=}6$) | 0.84 | 0.59 | 0.499 / 0.502 | 1% |
| temporal (64E, $w{=}6$) | 0.60 | 0.68 | 0.501 / 0.501 | 85% |

Every row of the table carries its own measured chance floor (token / context columns): the same
classifiers refit on null labels, using both iid permutations and circular shifts of at least
1000 tokens, the latter preserving the labels' residency-induced autocorrelation. All floors are
$0.500 \pm 0.002$, so chance is exactly 0.5 and every entry is real signal. The unconstrained router implements a near deterministic token-to-expert lookup: the
current token predicts expert firing at AUC 0.84 to 0.94 (+0.34 to +0.44 above floor), and
context never overtakes it for more than 1% of experts. The temporal model cannot implement that
lookup, because a token must be served by whichever experts are resident, and its token AUC
falls to 0.60 to 0.62 at both scales. Against the floor this is a reduction, not an erasure:
temporal experts retain a weak but genuine lexical signal (+0.10 to +0.12), roughly a quarter of
the unconstrained model's, while their contextual signal (+0.26) clearly dominates it. Freed
from token identity, temporal experts become context predictable: 85 to 91% are better predicted
by their surroundings than by the token they process, with the effect growing monotonically with
depth. We call this *de-lexicalization*. Context is the transferable
feature, autocorrelated within documents and shared across surface forms, and we identify this as
the regularization behind the loss improvement at scale. It equally explains why temporal routing
demand is far more predictable from history (AUC 0.85 vs 0.64 in our demand forecasting
analysis): context persists across neighboring tokens, token identity does not. The result is
robust to the window choice. Across a sweep $w \in \{\lfloor k/2 \rfloor, k, 32\}$ the fine pair
is essentially flat (context dominated 88 to 94%), and the coarse pair is context dominant at
both residency-scale windows (85 to 86%, with context AUC peaking at $w = k$), washing out to
balanced only at $w = 32$, a window five times its cache lifetime that dilutes the near context
carrying the signal. The coarse model's context specialization lives within one residency
lifetime, exactly where the cache can exploit it. Token AUC is window independent by
construction.

## Reading experts through the output vocabulary (P4)

The two analyses above examine what makes an expert fire. As an independent check we examine what
an expert *writes*. For each expert, average its actual output vectors over the tokens routed to
it, $v_e = \mathbb{E}_{t : e \in \mathrm{TopK}(t)}[\,\mathrm{out}_e(t)\,]$, project through the
final norm and unembedding $U$, and form $p_e = \mathrm{softmax}(U \cdot \mathrm{norm}(v_e))$,
the vocabulary distribution the expert promotes on its real traffic. Its sharpness is the
effective vocabulary $V_{\mathrm{eff}}(e) = \exp H(p_e)$, ranging from 1 (a single word) to the
vocabulary size (16k here, no lexical preference at all). A lexical expert should be readable as
a word cluster (low $V_{\mathrm{eff}}$, and its promoted words should overlap its trigger
tokens), while a contextual expert should be diffuse.

Data weighting matters: projecting raw weight columns with uniform weights yields
$V_{\mathrm{eff}} \approx 15{,}990$ for every expert in *both* models, because averaging
unactivated columns cancels their directions and mid-network projections are rotated relative to
the output basis. We take this unconditioned value as the no-signal reference: a projection
carrying no lexical preference at all reads about 15,990 of 16,000. All comparisons are
within layer and data weighted.

| model (192E) | mean $V_{\mathrm{eff}}$ | sharpest decile | no-signal reference |
|---|---|---|---|
| unconstrained | 15,439 | 13,431 | 15,990 |
| temporal | 15,932 | 15,342 | 15,990 |

Against the reference column, unconstrained experts promote measurably narrower vocabularies
(about 550 effective words of lexical preference at the mean, over 2,500 in the sharpest decile,
whose top promoted tokens read as coherent lexical clusters), while temporal experts are barely
distinguishable from no signal (about 60 at the mean) and contain no word-list experts even in
their extreme tail. One expert's write is a nudge rather than a full prediction, so the shifts
are small in absolute terms, but the direction agrees with both preceding analyses. Input side, output side, and structure thus converge on a single
mechanism: the residency constraint removes the router's lexical shortcut, and the experts
reorganize around context.

## Consequences: the mechanism is load-bearing at inference

If de-lexicalization were a pure training-time regularizer, the residency mask should be
removable at inference. We test this with a constraint swap: evaluate every trained model under
the *other* regime, using each pair's native evaluation protocol. Removing the mask from a
temporal model is implemented by setting the residency cache to the full pool (every expert
always resident, selection unconstrained), and the converse imposes the rolling-residency
mechanism on an unconstrained checkpoint.

| trained model | native loss | cross regime | delta |
|---|---|---|---|
| temporal, 192E at 1e16 | 1.4750 (masked) | 1.5744 (unmasked) | +0.10 |
| temporal, 64E at 1e17 | 1.2821 (masked) | 1.4063 (unmasked) | +0.12 |
| temporal, 1e18 (the winning case) | 3.9037 (masked) | 4.3890 (unmasked) | +0.49 |
| unconstrained, 192E at 1e16 | 1.4499 | 1.6902 (imposed) | +0.24 |
| unconstrained, 64E at 1e17 | 1.2690 | 1.8789 (imposed) | +0.61 |

Both directions hurt, so neither regime transfers: the advantage is serving co-adaptation, not
better weights. Even at 1e18, where the temporal model wins, unmasking it collapses the model
below both its masked self and the unconstrained baseline. The asymmetry makes the mechanism
causal rather than correlational: imposing residency on lexical routers costs two to five times
more than unmasking contextual ones, exactly as the locus analysis predicts, since a token
denied its bespoke expert has nowhere to go, while a contextual expert serves its neighborhood
regardless.

Finally, the constraint admits a dose. Decouple the cache size $R$ from the active top-k: the
cache holds $R$ experts (cold filled with the top $R$, evolving under the same one-swap-per-token
dynamics) and the router selects its top-k among residents, so $R = k$ is the maximal constraint
studied above, $R = E$ recovers the unconstrained model, and FLOPs are identical at every $R$.
Training from scratch at 1e16 (192 experts, k = 18) across the dose:

| $R$ | 18 (= k) | 36 | 72 | 128 | 192 (= E) |
|---|---|---|---|---|---|
| test BPB | 1.4750 | 1.4736 | 1.4681 | 1.4580 | 1.4475 |
| effective experts | 183.9 | 183.4 | 181.8 | 186.4 | n/a |

Loss falls monotonically as the constraint loosens, confirming that at this scale the constraint
is a pure quality cost whose regularization pays only at larger budgets, and expert diversity is
preserved at every dose, so the constraint acts on usage, never on expert identity. Because $R$
is the number of experts held in fast memory and compute is fixed, this curve *is* the serving
memory-quality frontier: the maximal constraint costs +0.028 BPB at roughly one tenth of the
routed-expert memory, and a system can buy back about a quarter of that gap by quadrupling the
cache.

## Appendix: an optimization control (P3)

A mundane alternative explanation is optimization noise: if temporal sequences use fewer distinct
experts, each expert might receive more tokens per update, giving lower gradient variance and
better training without any representational change. The temporal model does use fewer distinct
experts per sequence (158 vs 192 at the fine scale). However the per-expert update size is fixed
by the architecture: with top-k routing, every batch assigns exactly $k/E$ of its tokens to each
expert on average, independent of how assignments are distributed in time. Measured
tokens-per-expert-per-batch are identical across regimes (12,288 at the fine scale, 3,072 at the
coarse). The hypothesis is rejected, and the loss improvement cannot be attributed to gradient
batching.
