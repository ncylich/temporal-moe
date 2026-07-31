# Outstanding mechinterp work

**Scope.** This file answers one question only: *which outstanding runs need training and which do not.*
It is not a priority order. For sequencing see §4–§5 of
[`LAYER_LEXICALITY_ROUND2.md`](docs/research/mechanism/LAYER_LEXICALITY_ROUND2.md), which queues the
eval-only work ahead of the training tests; where the two disagree on what to do next, that document
wins.

Everything listed here was deferred during the layer-lexicality / mechinterp re-run on the grounds that
it needed artifacts fetched from Hugging Face. **That was wrong.** Every checkpoint, router log and
corpus shard in `results/MANIFEST.csv` is already on this machine, under a sibling checkout:

```
/workspace/FLAME-MoE/results/phase0/runs/     237 GB — all 72 runs, 69 with checkpoints (188 .distcp),
                                             22 router_log.pt, 3 delex_capture.pt, 9 act_log.pt
/workspace/FLAME-MoE/data/dclm_tokenized/      23 GB — 50k-vocab corpus (1e18, 1e19)
/workspace/FLAME-MoE/data/tok16k_full/         13 GB — 16k-vocab corpus (1e16, 1e17), complete
```

The run set matches `MANIFEST.csv` exactly. The Hugging Face cache holds only the `pythia-12b`
tokenizer — no checkpoints. Roughly 28 GB was re-downloaded into `temporal-moe/` for no reason, because
`analysis/paths.py` resolves `RUNS` inside whichever checkout you are in and I did not check the sibling.

**Point the tooling at the existing tree instead of downloading:**

```bash
export CKPT_ROOT=/workspace/FLAME-MoE/results/phase0/runs
export DATA_DIR=/workspace/FLAME-MoE/data/dclm_tokenized     # or data/tok16k_full at 1e16/1e17
export PY=/workspace/FLAME-MoE/.venv/bin/python               # the only venv with TransformerEngine
```

`PY` matters: `/workspace/FLAME-MoE/.venv` has the full pinned stack (torch 2.4.1+cu124,
TransformerEngine 1.11.0+fc034785, flash-attn 2.6.3, apex). `temporal-moe/.venv` has torch and sklearn
but no TE, so it cannot run a capture or an eval pass.

---

## 1. No retraining required — checkpoint is on disk

> **STATUS: all nine items complete.** Verified against artifacts on disk by
> `analysis/todo_status.py`, which checks contents — run counts, columns, arm sets — rather than file
> existence, because every failure on this branch came from a claim about "done" that a weaker check
> would have passed. Run it to re-confirm; it prints an explicit complete/outstanding line. The
> per-item text below is kept as the record of what each item was.

### 1a. Capture sweep (re-run plan Step 3) — 21 cells outstanding of 25

One forward pass each over the fixed 64×2048 batch, single GPU, a few minutes per run.
`scripts/phase0/delex_capture_sweep.sh --list` prints the set. Feeds the whole A-family
(locus, floors, lens, structural, demand, oracle, C9, C10).

| budget | runs still needing a capture |
|---|---|
| 1e18 | `flame38m_g5_moe`, `flame38m_g5_temporal`, `flame512_g1_moe`, `flame512_g1_temporal`, `flame512_g3_moe`, `flame512_g3_temporal`, `flame192_g3_moe`, `flame192_g3_temporal` |
| 1e19 | `moe_coarse_1e19`, `g1_tmoe_coarse_1e19`, `temporal_fine_g3_1e19` — **re-capture**, the existing three predate the layer-keying fix |
| 1e17 | `g3_moe_s1_1e17`, `g3_moe_s2_1e17`, `g3_moe_s3_1e17`, `g3_tmoe_s2_1e17`, `g3_tmoe_s3_1e17` |
| 1e16 | `g3_moe_s0_1e16`, `g3_moe_s1_1e16`, `g3_moe_sm1_1e16`, `g3_tmoe_s0_1e16_mom`, `g3_tmoe_sm1_1e16` |

Already captured post-fix, nothing to do: `flame38m_g1_temporal`, `flame38m_g1_moe`,
`flame38m_g3_temporal`, `flame38m_g3_moe` (in `temporal-moe/results/phase0/runs/`).

**Highest value in this block:** `g3_moe_s0_1e16`. It is the run behind the `s0_SOFTMAX_BASELINE` locus
row, the only cell ever measured at w=32 alone, which is why it is drawn dashed in `locus_by_layer.png`.
One capture pass closes that gap. Pair it with `g3_moe_s0_1e16_sigmoid_seed2` as the sigmoid control.

**Note on the 1e19 re-capture:** the three preserved captures have expert outputs keyed one layer too
shallow (`out_cnt is None` on the deepest layer), so every output-lens number derived from them is
misattributed and layer 14 was never covered. Fixed in `delex_probe.py`; the capture now refuses to write
a misaligned file. Routing metrics from those captures are unaffected and do not need redoing.

### 1b. A8 — weight geometry per layer

CPU only, no GPU, no forward pass: reads expert weights straight out of the checkpoint via
`ckpt_read.py`. `mechinterp_structural_1e19.csv` currently has `dist2centroid_mean`,
`pairwise_cos_med` and `pairwise_cos_p99` blank on all 39 rows with `geometry_note` explaining why.
I recorded the reason as "needs ~53 GB of 1e19 checkpoints"; they are present at 17 GB each
(`moe_coarse_1e19`, `g1_tmoe_coarse_1e19`, `temporal_fine_g3_1e19`). That reason was simply false.

### 1c. C5 — output lens beyond 1e18

Done at 1e18 (layers 2–9, all four arms). Needs the 1e19 re-captures from 1b/1a before it can extend
there. Checkpoints present.

### 1d. X3 — residency dose curve at 1e17 and 1e18

Evaluation only, no training: sweep `TEMPORAL_RESIDENCY_R` endpoints on an existing checkpoint. The
published dose curve covers 1e16 only. `scripts/phase0/constraint_swap_sweep.sh` already drives the
per-layer version; a uniform-R sweep is the same machinery with `TEMPORAL_R_SCHEDULE` unset.
**This item was dropped silently — it is in the plan's X family beside X1/X2 and I never mentioned it.**

### 1e. C8 — causal token / context substitution

Forward passes only, no training. The plan calls it *"the strongest non-training evidence available for
H1"* and it is the largest remaining gap in the C series. Needs new code — no script exists — plus a
GPU. Was never a download problem.

### 1f. e8 — document-boundary churn

I recorded this as permanently unrecoverable. **Also wrong.** It needs
`results/phase0/probe_batch_cache/eod_{16k,50k}.npy`, a `[B,S]` boolean mask of end-of-document
positions on the fixed eval batch. No committed code produces it — `probe_replay.py` only reads it — so
it needs a small new script, but the corpus it derives from is present in both tokenizations. New code,
not new training, and not unrecoverable.

### 1g. A11 — free-rider stats across all models

`mechinterp_freerider.csv` still has 4 rows carrying the old undecodable labels. I asserted it was
covered by `e2_streamed_diversity.csv` plus Appendix A rather than regenerating it. Half true:
tokens-per-expert is architecturally fixed (12,288 fine / 3,072 coarse, both regimes) and e2 gives
distinct-experts-per-sequence for 22 runs, but the file itself was never refreshed.

### 1h. `plot_probe.py` was broken — regression I introduced, now fixed

Not a plan item. Replacing the `model` column with `run,budget,regime,grain` in the replay CSVs broke
`plot_probe.py` (`KeyError: 'model'`), and `docs/ENVIRONMENT.md` guarantees all eleven plot scripts run
under `setup.sh analysis`. The other ten still pass. Fix the column read or add a compatibility shim.

### 1i. Two overstated claims to correct in the docs

- `MECHINTERP_RERUN_PLAN.md` §1 and §7.5 say the 1e16/1e17 locus cells "cannot be extended past layer 6,
  re-split, or re-windowed by anyone, ever." True for four of the five runs; **false for
  `g3_moe_s0_1e16`**, which is on disk with a checkpoint.
- The same sections describe e8 as unrebuildable from published artifacts. See 1f.

---

## 2. Requires retraining — no checkpoint exists anywhere

Verified absent from `/workspace/*/results/phase0/runs/` and from `MANIFEST.csv`.

### 2a. Published locus cells with no surviving checkpoint

| cell | run | granularity | substitute on disk? |
|---|---|---|---|
| `s0_TEMPORAL` @1e16 | `g3_tmoe_s0_1e16` | fine 18/192 | **no.** All 17 surviving `g3_tmoe_s0_1e16_*` runs are trigger-shaping variants (momentum, anti-pinning, bursty, head). Those knobs alter residency dynamics *during training*, so they are different trained models, not substitutes for the plain recipe |
| `s0_FULL` @1e16 | `g3_moe_s0_1e16_sigmoid` | fine 18/192 | **partially** — `g3_moe_s0_1e16_sigmoid_seed2` is the same recipe at another seed, so retraining is optional if a seed difference is acceptable |
| `s2_TEMPORAL` @1e17 | `tmoe_minlogit_sh1_s2_1e17` | **coarse 6/64** | **no** |
| `s2_FULL` @1e17 | `v16k_sweep_s2_1e17` | **coarse 6/64** | **no** |

**The coarse 1e17 pair is the real gap.** Every 1e16 and 1e17 run on this machine is grain=3, fine
18/192 — there is no coarse 6/64 checkpoint at either budget. That pair cannot be recovered by any
amount of inference.

**Worth weighing before spending on it.** These models are 4–6 layers deep, so "full depth" is 3–5 MoE
layers, and that is exactly where curvature intervals were measured straddling zero — the depth-shape
question cannot be answered at that depth however good the checkpoint is. What retraining buys is the
cross-budget *level* comparison, which the surviving fine 1e16/1e17 runs already supply.

### 2b. Runs behind the published e1–e8 replay numbers

All absent everywhere: `tmoe_minlogit_sh1_s0_1e16`, `tmoe_minlogit_sh1_s2_1e17`,
`tmoe_minlogit_sh1_s3_1e17`, `g3_tmoe_s1_1e17`, `flame38m_temporal_minlogit`, and the matched full-MoE
runs `v16k_d_s0_1e16`, `v16k_sweep_s2_1e17`, `v16k_sweep_s3_1e17`.

The e1–e8 re-run already **replaces** these over the 22 preserved logs, so retraining is only needed to
reproduce the *published* numbers as such, not to have the metrics. Low priority.

### 2c. T1 / T2 / T3 — the H2 training tests

**Not to be started without a decision.** C3 has run and its per-layer cost profile is U-shaped
(vertex layer 5.3 unmasking, 5.5 imposing; ends ÷ middle 1.40× and 1.52×), which falsifies H2 on its own
pre-registered criterion. It also mis-specifies T2: T2 contrasts shallow-half against deep-half, which
splits the U through its minimum and would return a null whatever the truth. Any redesign should contrast
ends against middle — {2,3,8,9} versus {4,5,6,7} at 1e18 — at matched layer count and resident-slot
budget. See `LAYER_LEXICALITY.md` §3 and §5.

---

## 3. Not done, and why

Kept separate from §2 because none of this needs training. It is work that was identified, scoped and
deliberately not finished, recorded so the next person does not rediscover it or assume it was
overlooked.

**Current state:** 3a, 3f and 3g are **resolved** and kept for the record of what was wrong and how it
was found. 3d is **not a bug** — it is recorded because it was wrongly escalated into one. 3b, 3c, 3e
and 3i remain genuinely open. 3h is out of scope by instruction.

### 3a. In-process sweep evaluation — **fixed and validated**

`analysis/probes/sweep_eval.py` loads the model once and loops residency settings in-process, avoiding
the ~4 min of Megatron/TE init, dataset index build and checkpoint load that each separate arm pays.
It reproduces independently-measured references to **1e-6**:

| arm | in-process | per-arm reference |
|---|---|---|
| dose_R24 | 4.102363 | 4.102362 |
| dose_R48 | 4.316587 | 4.316586 |
| dose_R64 | 4.388953 | 4.388952 |

Validated a second time on a different model with per-layer schedules rather than uniform R
(`flame38m_g3_temporal`, five arms, ~2e-6). Measured effect: five arms in ~10 min against ~10 min
*each* for the per-process path.

**It was broken in three independent ways, each of which produced plausible output:**

1. **The temporal router was never installed.** The script called `pretrain()` on `pretrain_gpt`
   directly instead of patching `TopKRouter.forward` first, as `temporal/pretrain_temporal.py` does.
   The model ran as a plain MoE and the residency env vars were read by code that never executed, so
   every arm returned the same loss — a flat dose curve, which is a *result-shaped* failure, not a
   crash.
2. **The cache held 1/32 of the reference data.** The iterator yields micro-batches but an eval
   iteration consumes a whole global batch, so caching `eval_iters` micro-batches scored on 16 where
   the reference used 512.
3. **The selftest demanded 1e-9 agreement between repeated GPU passes**, which are not bitwise
   deterministic. It failed at 3.87e-08 — a correct result rejected by an unmeetable check.

**The check that was missing, now added:** distinct R values must produce distinct losses. The
original selftest only verified that a *repeated* arm saw the same batches, so it was structurally
incapable of noticing that every arm was identical to every other — the actual failure.

### 3b. Eval volume — two cuts are free, the third is not

Each eval arm pushes **~88M tokens to produce one scalar**, at a measured 226k tokens/s and 40
TFLOP/s. Throughput is not the problem; volume is. Of that: 21M tokens are frozen "training"
(forward *and* backward) at `lr=0`, and ~33M are a validation set that is computed and discarded —
only the test number is read. `eval_iters=16` at global batch 1024 x 2048 is roughly 16x more data
than a stable CE needs; val and test already agree to 0.0008 nats, which is the signature of being
far past diminishing returns.

Three cuts are available, and they differ in whether they change the measured number:

| cut | saving/arm | changes the number? | evidence |
|---|---|---|---|
| drop frozen train iters (`EVAL_TRAIN_ITERS`, already wired into `run.sh`) | ~1.5 min | **no** | `sweep_eval` ran at `--train-iters 1` against references at 10 and reproduced `dose_R24/R48/R64` to **1e-6**. At `lr=0` the weights do not move. |
| skip the validation pass | ~0.8 min | **no** (one confirmation run to be sure) | separate iterator from test; only the test number is ever read |
| `eval_iters` 16 → 2 | ~0.7 min | **yes** | at 160 micro-batches the sweep gave 4.099577 against the reference 4.102362 — off by 0.0028, the same order as the effects being resolved. Matched to 1e-6 only at the full 512. |

**Apply the first two and not the third: they are free, already evidenced, and worth ~30% per arm,
whereas the third buys a further ~10% at the price of making every new arm incomparable to the ~60
already measured.** All of it is small next to the in-process sweep (3a), which amortises the ~4 min
startup that actually dominates an arm — that is the 5x, and it has landed.

### 3c. Parallelism not yet applied to `delex_lens`, `delex_structural`, `delex_demand`

The pool pattern is proven on two analyses and both were verified equivalent, not merely faster:

| analysis | before | after | verification |
|---|---|---|---|
| `delex_locus_driver` | 9143 s | 1288 s (7.1x) | all 87552 rows matched by key, median diff 0.000000 |
| `delex_oracle` | part of a 2032 s block | 53 s | all 29184 rows matched, `n_token_ids` identical in all 26 runs |

The remaining three share the same `for r in cells` shape. `delex_lens` has two row-append sites
rather than one, so it needs slightly more care than a mechanical copy of the patch. Use the same
acceptance test: save the serial CSV, run parallel, diff by key — identical rows, not wall-clock.

### 3d. Serial/parallel numeric difference — **not an issue; I over-escalated it**

Parallel and serial locus outputs differ at the 1e-3 level: median exactly 0.000000, 60% of values
bit-identical, p99 0.0007, max 0.0034. Effect sizes being resolved are ~0.03 nats, so this is a 10x
margin. **This is ordinary floating-point non-determinism and does not affect any conclusion.** The
correct handling is a line in the commit message saying the outputs are equivalent within tolerance.

Recorded here only because I originally wrote it up as an open investigation, which was wrong twice
over:

1. It never needed investigating. Non-associative summation is a known property of parallel numerics
   and ML results are not sensitive at this magnitude.
2. The mechanism I proposed was also false. I attributed it to BLAS reduction order from capping
   workers at 8 threads against the serial run's 208. Running the same capture serially at both thread
   counts gives **100% bit-identical output** (4608/4608 rows), on a capture that does differ between
   serial and parallel. Thread count is not the cause.

The one place numerical noise could bite is a gate whose tolerance sits at the same scale, and the
null gate's is +-0.002. But that framing inverts the priority: the gate's own estimator carried
+-0.002 of sampling noise from `max_experts=24`, which is what actually flagged four healthy models
(see 3f). That was the real defect and it is fixed. The residual reduction-order noise is immaterial
beside it.

If anyone does want the remaining curiosity closed: the difference is against a baseline captured at
18:13 on 07-30 by the pre-parallel driver, and `delex_locus.py` last changed at 00:59 that day, so
code drift is ruled out too. A pool-versus-no-pool run on identical code isolates the process
boundary. Not worth compute unless something else makes it interesting.

### 3e. `input_ids` not added to the capture writer

`eod_capture.py` loads a full model — checkpoint, dataset index, ~5 minutes of GPU — purely to read
back **input token IDs**, discarding the model's output entirely. Those IDs are the same fixed batch
`delex_probe.py` and `router_probe.py` already push through when they write the capture. Recording
`input_ids` alongside the router logits would make 1f pure post-processing: no GPU, no model, seconds.

It would also have prevented both of this item's failures, since neither is possible without a model
load: the tokenizer `eod_id` lookup raising on `_HuggingFaceTokenizer`, and the 192-expert checkpoint
being built as 64 experts because `GRAIN=3` was omitted.

Not done because the existing captures have no `input_ids` field, so it only helps future captures
unless all 26 are re-run — which was not worth the GPU time once the mask existed.

### 3f. Four models flagged by the null battery — **resolved: the test's own noise**

Four of 26 fell outside median iid-null AUC 0.500 ± 0.002. Re-running the identical arm with the
expert cap raised from 24 to 256:

| run | dev at n=24 | dev at n=256 |
|---|---|---|
| `flame38m_g1_moe` | 0.0020 | 0.0002 |
| `flame512_g1_temporal` | 0.0021 | 0.0001 |
| `g3_moe_s0_1e16_sigmoid_seed2` | 0.0025 | 0.0009 |
| `g3_tmoe_s0_1e16_mom` | 0.0025 | 0.0004 |

Every deviation shrinks 3–20×, as a median genuinely centred on 0.500 should. All 26 runs now pass at
the raised default (worst deviation 0.0012). **No model's numbers are withdrawn.**

The defect was in the test: `max_experts=24` gives the median about ±0.002 of sampling noise — *the
same size as the 0.002 tolerance it is compared against*. A gate whose noise floor equals its
threshold flags healthy models at a steady rate. Default raised to 256.

### 3g. L3 disagreement — **resolved: real, not sampling noise**

Exempting layer 3 costs +0.068 nats on the granularity-variation model against +0.039 on the
seed-variation model, while every other interior layer agrees between the two to within 0.006. With one
run per cell there was no error bar, so this could be neither claimed nor dismissed.

Tested by re-running native and L2-L5 on `flame38m_g3_temporal` across two independent evaluation data
draws (seeds 1234 and 4321). Per-layer costs against each draw's own native:

| layer | seed 1234 | seed 4321 | difference |
|---|---|---|---|
| L2 | +0.036192 | +0.036187 | 5e-6 |
| L3 | +0.067568 | +0.067727 | 1.6e-4 |
| L4 | +0.038897 | +0.038894 | 3e-6 |
| L5 | +0.032965 | +0.033136 | 1.7e-4 |

Absolute losses moved with the data as expected (native 3.9769 -> 3.9709), but the costs are stable to
about 1e-4 -- three orders of magnitude below L3's 0.031 excess over its neighbours. L3 is 1.8x its
neighbours in both draws.

**Conclusion:** the L3 spike is a real property of this model, so the gap against the seed-variation
model is a genuine model-level difference rather than measurement noise.

**What this does NOT establish:** it varies the evaluation draw, not the training seed, so it cannot
separate "granularity produces an L3 spike" from "this particular model has one". That needs a second
g3 model at a different training seed, which is a training run -- see section 2.

The seed-1234 arm also reproduced the original per-arm measurements to about 2e-6 on all five points,
which is a second independent validation of the repaired in-process sweep (3a) on a different model and
with per-layer schedules rather than uniform R.

### 3i. Subset runs silently truncate the full results CSV — **open trap**

`delex_locus_driver.py`, `delex_oracle.py` and their siblings **rewrite their entire output CSV from
whatever cells they were given**, rather than merging into what is already there. Running one on a
named subset for a quick diagnostic therefore replaces the full result set with the subset. It bit me
once: a two-run pool-versus-no-pool comparison cut `mechinterp_locus_1e19.csv` from 87552 rows over 26
runs to 9216 over 2. Recovered from the committed version with no loss, because it was committed.

This is fine when a full regeneration is intended and dangerous otherwise, and nothing warns about it.
The fix is either to merge on `(run, layer, expert, …)` rather than overwrite, or to refuse to write a
file with fewer runs than the one on disk unless an explicit `--replace` flag is passed. Neither is
done.

Until then: **commit before running any of these on a subset.** That is the only reason the incident
above cost nothing.

### 3h. T1–T4 — deliberately not started

Out of scope by explicit instruction. Everything they need is in §2.

---

## 4. Audit log

Each entry: what was audited, how, what was found, what was fixed. A pass that finds nothing is still
recorded — it tells the next person that class was checked and when, so it does not get redone.

### 2026-07-31 — six-pass fresh-context audit of the doc set

Scope: `docs/research/mechanism/*.md`, `TODO.md`, `README.md`, against `results/ablations/*.csv`.
Method: six independent Sonnet subagents with **disjoint scopes**, briefed with the artifacts only —
no narrative, no conclusions, no history, since a briefed auditor confirms the briefer's framing.
Passes: numbers, status, cross-document, superseded, reproduction, links.

**Found and fixed, by class rather than by instance:**

| class | instances found | instances fixed |
|---|---|---|
| stale `file.py:NN` line pins | 3 (2 broken, 1 correct-by-luck) | 3 — all converted to symbol references, which break only on a rename |
| detail text left stale when a status line was updated | 12 across 5 files | 12 |
| "cannot be extended past layer 6 by anyone" — false for `g3_moe_s0_1e16` | 3 passages | 3 |
| hand-maintained coverage counts drifting from their CSVs | 6 rows | replaced with generation (`analysis/coverage_table.py`) |
| unlabelled metric (val CE quoted beside test CE) | 1 | 1 |
| corrections identified elsewhere but never applied | 2 | 2 |
| broken cross-document link after a file move | 1 | 1 |
| miscounted artifacts (PNGs, `.distcp`, disk size) | 3 | 3 |

**Checked and rejected — recorded because a fresh reader tripping on correct text is also signal:**

- *"README's Pixel 10a claim has no supporting evidence"* — false positive caused by the **brief**, not
  the doc. The auditor was scoped to `results/ablations/`; the evidence is in `androidbench/`. Scope a
  numeric audit to every directory that could hold evidence, or expect this class of false positive.
- *"`delexicalization.md` reports native CE 3.9037 against 3.909461 in the CSV"* — a real conflict with
  the wrong diagnosis. 3.9037 is `val_CE` from `unmask_eval.csv` and its partner 4.3890 is the same
  row, so the table is internally consistent; the defect is an **unlabelled metric** sitting beside
  test-CE figures. Fixed by labelling, not by changing the value.
- *"17% decode slowdown should be 18%"* — rounding convention, and a different candidate row gives 12%.
  The claim is not wrong; which row is canonical is ambiguous. Left alone.
- *"`TODO.md` §1's per-item text describes outstanding cells"* — one auditor flagged it, another
  explicitly exonerated it as adequately marked by the section header. The header is adequate, but two
  of three fresh readers tripping on it suggests the marker could be stronger.

**Verified but unresolved:** the depth-slope CI bounds in prose differ from `mechinterp_locus_slopes.csv`
by 0.0004–0.0026 while every point estimate matches exactly. The auditor attributed this to an unseeded
bootstrap; `delex_locus.py` seeds with `default_rng(0)`, so that explanation is wrong and the cause is
unidentified. Recorded rather than guessed at.

**Caveat on the method itself:** running several agents with git access in one working tree caused
`.git/index.lock` contention, and one pass runs `git checkout -- .` as cleanup, which would discard
another agent's uncommitted work. Commit before starting a sweep, and expect to wait on the lock.
