# Why the Temporal Constraint Wins: De-Lexicalization of Routing

Rolling residency restricts each token to the currently resident expert set, yet at 1e18 FLOPs
the temporally constrained model achieves *lower* validation loss than an unconstrained MoE of
identical architecture and compute, at both granularities (coarse: CE 3.9121 vs 3.9209, fine:
3.9768 vs 4.0087, 50k vocabulary; a 1e19 replication is in progress). A constraint that improves
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
| temporal (192E) | 0.66 | 65 | 0.95 | 0.88 | 0.010 |
| unconstrained (192E) | 0.25 to 0.28 | 5 to 12 | 0.85 to 0.87 | 0.88 to 0.90 | 0.006 to 0.010 |
| temporal (64E) | 0.52 | 54 | 0.93 | 0.84 | 0.012 |
| unconstrained (64E) | 0.34 | 13 | 0.86 | 0.84 | 0.009 |

The weight geometry is indistinguishable across regimes: experts remain equally distinct and
near orthogonal either way. What changes is traffic. Unconstrained experts each draw their usage
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

| model | median $A_{\mathrm{tok}}$ | median $A_{\mathrm{ctx}}$ | context dominated |
|---|---|---|---|
| unconstrained (192E, $w{=}18$) | 0.94 | 0.63 | 0% |
| temporal (192E, $w{=}18$) | 0.62 | 0.77 | 91% |
| unconstrained (64E, $w{=}6$) | 0.84 | 0.59 | 1% |
| temporal (64E, $w{=}6$) | 0.60 | 0.68 | 85% |

The unconstrained router implements a near deterministic token-to-expert lookup: the current
token predicts expert firing at AUC 0.84 to 0.94, and context never overtakes it for more than
1% of experts. The temporal model cannot implement that lookup, because a token must be served
by whichever experts are resident, and its token AUC collapses to 0.60 to 0.62 at both scales.
Freed from token identity, its experts become context predictable: 85 to 91% of temporal experts
are better predicted by their surroundings than by the token they process, with the effect
growing monotonically with depth. We call this *de-lexicalization*. Context is the transferable
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

Data weighting matters: projecting raw weight columns with uniform weights yields near uniform
distributions for every expert in both models, because averaging unactivated columns cancels
their directions and mid-network projections are rotated relative to the output basis. All
comparisons are therefore within layer and data weighted.

Unconstrained experts promote measurably narrower vocabularies, mean $V_{\mathrm{eff}}$ 15,439
vs 15,932 for temporal, and the contrast concentrates in the tail: the sharpest decile of
unconstrained experts reaches $V_{\mathrm{eff}} = 13{,}431$, and their top promoted tokens read
as coherent lexical clusters, while the sharpest temporal decile only reaches 15,342, so the
temporal model contains no word-list experts even in its extreme tail. The absolute shifts are
small, as one expert's write is a nudge rather than a full prediction, but the direction agrees
with both preceding analyses. Input side, output side, and structure thus converge on a single
mechanism: the residency constraint removes the router's lexical shortcut, and the experts
reorganize around context.

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
