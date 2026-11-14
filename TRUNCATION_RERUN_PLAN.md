# Truncation re-run plan

Status: proposed, not started. Owner: trajectory re-run track.
All numbers below were computed from the committed dumps on 2026-08-23 and are reproducible
with the snippets in each section.

## 1. Why this exists

The recovered per-item dumps show that most of the "blown-up generations are wrong" effect is
generations getting cut off at the token cap, not generations degrading. Pooled over the 31
recovered cells, splitting every blown-up generation by how it actually ended:

| Group of constrained generations | HumanEval n / wrongness | MMLU n / wrongness |
|---|---|---|
| A. Hit the cap, never closed the thinking block | 139 / 0.978 | 143 / 0.434 |
| B. Hit the cap, thinking closed, answer cut off | 44 / 0.909 | 39 / 0.641 |
| C. Ran over 2x the free counterpart, finished cleanly | 113 / 0.150 | 286 / 0.262 |
| D. Blown only because the free run hit its cap | 50 / 0.100 | 85 / 0.118 |
| Normal length | 2278 / 0.046 | 2867 / 0.156 |

Groups A and B are mechanical: a generation that never emitted a program or an answer scores
wrong because it emitted nothing. Only group C speaks to a length-quality relationship, and it
is 3.3x on code and 1.7x on knowledge, not the 11.1x and 2.0x that pooling A, B and C together
produces.

The problem is that at the current cap we cannot tell whether group A and B generations would
have been right. Excluding them shifts measured damage by a median of 1.3 points but by up to
7.5 points on cells the paper builds claims on:

| Cell | Damage as reported | Damage excluding truncated | Shift |
|---|---|---|---|
| gemma4 think-on HumanEval, R=8 | -12.2 | -4.7 | +7.5 |
| Qwen3.5 think-on MMLU, R=8 | -6.6 | +0.8 | +7.4 |
| gpt-oss-20b high HumanEval, R=4 | -5.5 | -2.3 | +3.2 |
| gemma4 think-on MMLU, R=8 | -4.4 | -1.9 | +2.5 |

Section 6.1's claim that thinking-on roughly triples gemma4's damage, and the thinking
amplification story generally, currently rest on numbers where a third of the generations in
the worst cell never finished. That has to be resolved before those claims are restated.

## 2. Scope

Only generations that hit the cap need redoing. A generation that terminated naturally under
the old cap terminates identically under a larger one, because generation is prefix-determined
and the sampler is seeded, so leaving it untouched is equivalent to re-running it at zero cost.

**666 of 12,088 generations (5.5%)** hit the cap, across both arms of all 31 cells. They carry
2.43M of the grid's 10.84M decoded tokens.

| Option | Decoded tokens | Share of a full 2x re-run |
|---|---|---|
| Re-run the entire grid at 2x budget | 21.67M | 100% |
| Re-run only truncated generations, from scratch, at 2x | 4.87M | 22.5% |
| Continue only truncated generations for one more budget | 2.43M | 11.2% |

### Work list

Both arms must be redone wherever either arm truncates, because damage is a within-cell paired
contrast and is only valid at a matched cap.

| Surface | Cell | Current cap | Truncated free | Truncated constrained |
|---|---|---|---|---|
| MMLU | Qwen3.5 think-on, R=8 | 4096 | 63 | 78 |
| MMLU | Qwen3.5 think-on, R=32 | 4096 | 63 | 61 |
| HumanEval | gemma4 think-on, R=8 | 3072 | 35 | 50 |
| HumanEval | gemma4 think-on, R=16 | 3072 | 35 | 39 |
| HumanEval | gpt-oss-20b high, R=4 | 4096 | 21 | 27 |
| HumanEval | gpt-oss-120b high, R=4 | 4096 | 12 | 15 |
| HumanEval | gpt-oss-120b high, R=16 | 4096 | 12 | 9 |
| MMLU | gemma4 think-on, R=8 | 4096 | 7 | 14 |
| HumanEval | LFM2.5, R=4 | 4096 | 4 | 14 |
| MMLU | gemma4 think-on, R=16 | 4096 | 7 | 7 |
| MMLU | LFM2.5, R=4 | 4096 | 4 | 5 |
| HumanEval | gpt-oss-20b med, R=4 | 2048 | 6 | 2 |
| HumanEval | gemma4 think-off, R=8 | 1536 | 3 | 4 |
| HumanEval | Qwen3.5 think-on, R=8 | 4096 | 1 | 6 |
| HumanEval | gemma4 think-off, R=16 | 1536 | 3 | 3 |
| HumanEval | Qwen3.5 think-on, R=32 | 4096 | 1 | 5 |
| HumanEval | gpt-oss-120b med, R=4 | 2048 | 3 | 3 |
| HumanEval | gpt-oss-120b med, R=16 | 2048 | 3 | 3 |
| MMLU | gpt-oss-20b high, R=4 | 4096 | 3 | 3 |
| MMLU | gpt-oss-120b high, R=16 | 4096 | 2 | 4 |
| MMLU | gpt-oss-120b high, R=4 | 4096 | 2 | 3 |
| MMLU | gpt-oss-20b low, R=4 | 1727 | 2 | 1 |
| *(9 further cells at 1 or 2 generations each)* | | | | |

Regenerate the work list rather than trusting this table if the dumps change:

```
python analysis/residency/length_extension.py     # refreshes length_extension.csv
# then re-derive per-cell cap-hit counts from results/ablations/genbench_samples/
```

## 3. The two blockers

Continuing a truncated generation from where it stopped, rather than regenerating it, would
halve the cost of this work. Two things must hold for a continuation to be faithful, and
**neither holds today**. Both are fixable. Both need a test before any number produced by them
is trusted.

### Blocker A: the prefix cannot be reconstructed for half the models

The dumps store `raw` text and a token count. They do **not** store token IDs:

```
item keys: ['doc', 'gen_toks', 'gold', 'pred_relaxed', 'pred_strict', 'raw', 'text', 'think_toks']
```

So resuming means re-tokenizing the saved text, which must reproduce the original token
sequence exactly. Measured, 60 to 80 items per model:

| Model | Re-tokenized length matches recorded | Median drift | Worst |
|---|---|---|---|
| Qwen3.5 | 80 of 80 | 0 | 0 |
| gemma4 think-on | 8 of 60 | -1 | -11 |
| gpt-oss-120b high | 3 of 60 | -1 | -3 |
| LFM2.5 | 1 of 60 | -1 | -23 |

Qwen round-trips exactly. The three families whose transcripts carry channel and think markers
do not, because those markers were written into `raw` as literal text and do not re-tokenize
back to the same special-token IDs. Continuing from a prefix that differs by 1 to 23 tokens
from what the model actually produced is a silent corruption: the output looks entirely
plausible and is a continuation of a sequence the model never generated.

**Fix**

1. Generation drivers persist `prompt_ids` and `gen_ids` alongside `raw`. This is the durable
   fix and costs nothing at write time.
2. For dumps that already exist, a continuation is permitted only for (model, task) pairs that
   pass test T1 below. Everything else regenerates from scratch.

**Test T1, tokenizer round-trip gate**

New file `analysis/residency/tests/test_prefix_roundtrip.py`.

- For every committed dump, re-tokenize `raw` with that model's tokenizer and assert the
  resulting ID sequence, re-decoded, is byte-identical to `raw`, and that its length equals
  `gen_toks`.
- Assert on IDs, not on length alone. Equal length does not prove equal segmentation.
- Emit a machine-readable allowlist of (record, task) pairs that pass. The continuation path
  reads that allowlist and refuses anything absent from it.
- Expected today: Qwen passes, gemma4 and gpt-oss and LFM2.5 fail. The test is green when it
  reports that accurately, not when everything passes.

### Blocker B: the residency state cannot be resumed

`compute_resident_mask` (`temporal/temporal_router.py:39`) seeds the resident set from the
**first** token's top-k and walks forward one swap per token:

```
resident = torch.zeros(B, E, dtype=torch.bool, device=dev)
_, top_i = trig[0].topk(k, dim=-1)
resident.scatter_(1, top_i, True)
out[0] = resident
```

There is no argument to start from a given resident set, and S_t was never saved. Under the
decode-time protocol the prompt is processed freely and the constraint applies to generated
tokens, so a naive resume that feeds prompt plus generated-so-far as a new prompt would prefill
the whole prefix **free**. The resident set at the resumption point would be whatever free
routing yields at the last prefix token, not the state the constrained walk had reached after
thousands of constrained steps. The resident set is path-dependent by construction, so this is
not a continuation of the constrained run. It is a different run wearing its prefix.

**Fix**

1. Add an optional `init_resident: torch.Tensor | None = None` (and matching `init_refresh`) to
   `compute_resident_mask`, defaulting to today's cold fill so existing behaviour is unchanged.
2. Add a constrained teacher-forced replay that reconstructs S_t over a saved prefix. This is
   one forward pass, not N decode steps: within a layer the residency walk depends only on that
   layer's router logits for the whole sequence, and those logits depend on the layer input,
   which the already-constrained earlier layers produced. So the replay proceeds layer by layer
   exactly as constrained training already does.
3. Persist the resident set at the end of generation so future resumes need no replay at all.

**Test T2, resume equals continuous run, bit-exact**

New file `temporal/tests/test_resume_residency.py`, in the style of the existing
`test_temporal_router.py` and the walker and decode-accel tests, which already hold new paths
to bit equality against a reference.

- Draw random router logits of shape [S, B, E]. Run `compute_resident_mask` over the full
  sequence to get the reference mask.
- Split at every cut point t in a randomized set. Run the walk over [0, t), capture the state,
  then run over [t, S) seeded with `init_resident` and `init_refresh` from that capture.
- Assert the spliced mask is **bit-identical** to the reference over all S positions, not
  approximately equal.
- Cover the cases the existing router tests cover, since they are where the eviction order is
  ambiguous: exact logit ties, all-equal logits, R equal to E, R equal to 1, minus-infinity
  entries, and both `evict` modes.
- Then the end-to-end case: one constrained generation of N tokens, versus the same generation
  stopped at N/2 and resumed through the replay path. Assert identical output token IDs under
  a fixed seed.

Test T2 failing is a hard stop. It is the mechanism the paper is about, and a resume that
silently diverges would contaminate every number downstream of it.

## 4. Recommended path

**Regenerate the 666 truncated generations from scratch at double budget.** 4.87M tokens,
about 22% of a full re-run, no splicing caveat, and no dependency on either blocker.

Continuation saves a further 2.43M tokens, roughly 11% of a full re-run. That is not worth
gating this work behind a prefix reconstruction that is provably broken on three of six model
families plus new resume code on the hot path of the core mechanism.

Do the fixes anyway, on their own timeline. Token IDs in the dumps and a resumable residency
walk are both worth having: the first makes every future re-run cheap, the second is a
prerequisite for any long-context or interrupted-decode work. They just should not block this.

## 5. Re-run protocol

- Double the generation budget for both arms of each cell in the work list. Free and
  constrained must share an identical cap.
- Everything else unchanged: same sampling recipe, same seed, same items, same scoring, same
  residency setting.
- Per-item dumps on, with token IDs added per blocker A.
- Write to a separate CSV, adjudicate against the existing rows, then promote. Do not overwrite
  the authoritative file in place.

## 6. Acceptance criteria

1. T1 and T2 both present and passing, or the continuation path is not used at all.
2. Every re-run cell reports: truncation rate at the new budget, damage at the new budget, and
   damage at the old budget over the same items.
3. Truncation at the new budget is materially lower. If a cell still truncates above about 10%,
   doubling was not enough and that cell needs a further increase rather than a shrug.
4. The A / B / C / D decomposition in section 1 is recomputed at the new budget. The claim the
   paper will make is group C, so group C needs to survive with a sample worth quoting.
5. The MMLU three-way rescore (reported, honest, finished-only) is extended to HumanEval. This
   needs no GPU, the dumps are committed, and it gives the same decomposition on code.

## 7. Open question that gates the gpt-oss cells

`analysis/residency/mmlu_gptoss.py:124` chooses the sampling recipe by testing whether the
shipped `generation_config.json` contains `temperature`, `top_p` or `top_k`. gpt-oss-120b's
file contains none of them, so the harness takes the 0.7/0.95 no-recipe branch, and gpt-oss-20b
has no `generation_config.json` in the local snapshot at all. If gpt-oss's published guidance
does specify a recipe, that detection is looking in the wrong place and the superseded 1.0/1.0
rows were the correct ones.

Eight of the cells in the work list are gpt-oss. Settle this before spending GPU time on them,
because it decides which era of gpt-oss rows is authoritative and therefore what the re-run is
even being compared against.

## 8. What this unblocks in the paper

Section 6.2 currently fuses three claims that separate cleanly and survive at different
strengths. The re-run is what lets each be stated at its real strength:

1. The constraint pushes more generations into the budget wall. Cap-hits go from 142 to 183 on
   code and 159 to 182 on knowledge. Strong, and currently buried.
2. A generation that hits the wall almost always fails. True but mechanical, and it happens
   under free routing too, so it is a property of the budget rather than of residency.
3. Among generations that finish cleanly, unusually long ones are wronger. 3.3x on code and
   1.7x on knowledge. Modest, and the only part that is genuinely about derailment.
4. Wrong-flips exceed rescues 1.64:1 across all three surfaces (715 against 437). Untouched by
   any of this, and the asymmetry claim generalizes.

The sentence in Section 6.2 claiming the effect holds "in every thinking configuration under
both settings (20 of 20 comparisons)" is falsified by the fuller data at 23 of 30, and should
not be replaced with the new count. It should be rewritten around the layering above.
