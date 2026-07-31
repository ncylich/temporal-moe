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
/workspace/FLAME-MoE/results/phase0/runs/     201 GB — all 72 runs, 69 with checkpoints (178 .distcp),
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

### 1h. `plot_probe.py` is broken — regression I introduced

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

### 3a. In-process sweep evaluation — written, **fails validation, do not use**

`analysis/probes/sweep_eval.py` loads the model once and loops residency settings in-process, to avoid
paying ~4 min of Megatron/TE init, dataset index build and checkpoint load per arm. It does not work.
Three arms at R=24/48/64 returned an identical loss of 4.403726, against a measured `dose_R24` of
4.102362.

Cause: the script calls `pretrain()` on `pretrain_gpt` directly and never installs the temporal router
patch, so the model evaluates as a plain MoE and `TEMPORAL_RESIDENCY_R` is read by code that never
runs. The result lands near the unconstrained reference (4.3890), which is what an unpatched model
should give.

To finish: install the temporal patch the way the other entry points do, and match the cached batch
count to the reference config (it caches 20 micro-batches; the references use 16 eval iters at global
batch 1024, so losses are not comparable until that agrees). Acceptance is reproducing `dose_R24 =
4.102362`, not a speedup.

**Lesson worth keeping.** Its `SWEEP_SELFTEST` check passed. It only verified that a *repeated* arm
sees the same batches; it could not detect that all three arms were identical *to each other*, which
was the actual defect. The check that would have caught it is trivial: two different R values must
produce different losses. This is the same shape of error as the EOD mask verifier that compared
shapes when both tokenizers give (64, 2048) — a guard written against the imagined failure rather than
the real one.

### 3b. Eval volume — measured, not cut

Each eval arm pushes **~88M tokens to produce one scalar**, at a measured 226k tokens/s and 40
TFLOP/s. Throughput is not the problem; volume is. Of that: 21M tokens are frozen "training"
(forward *and* backward) at `lr=0`, and ~33M are a validation set that is computed and discarded —
only the test number is read. `eval_iters=16` at global batch 1024 x 2048 is roughly 16x more data
than a stable CE needs; val and test already agree to 0.0008 nats, which is the signature of being
far past diminishing returns.

Three cuts, worth ~3x per arm: drop the frozen train iters (`EVAL_TRAIN_ITERS`, already wired into
`experiments/run.sh`), skip the validation pass, and reduce `eval_iters` to ~2. Comparability holds as
long as every arm uses the same setting — which is exactly why this was **not** applied mid-programme:
the arms already measured would not be comparable to arms measured after the change. Do it at a clean
boundary and re-measure the reference points.

### 3c. Parallelism not yet applied to `delex_lens`, `delex_structural`, `delex_demand`

The pool pattern is proven on two analyses and both were verified equivalent, not merely faster:

| analysis | before | after | verification |
|---|---|---|---|
| `delex_locus_driver` | 9143 s | 1288 s (7.1x) | all 87552 rows matched by key, median diff 0.000000 |
| `delex_oracle` | part of a 2032 s block | 53 s | all 29184 rows matched, `n_token_ids` identical in all 26 runs |

The remaining three share the same `for r in cells` shape. `delex_lens` has two row-append sites
rather than one, so it needs slightly more care than a mechanical copy of the patch. Use the same
acceptance test: save the serial CSV, run parallel, diff by key — identical rows, not wall-clock.

### 3d. The BLAS-thread explanation is inferred, not demonstrated

Parallel and serial outputs differ at the 1e-3 level (median exactly 0, p99 0.0007, max 0.0034 on
locus). The explanation is floating-point reduction order: workers are capped at 8 BLAS threads where
the serial run used all 208. That is well supported but not proven. The clean control is a **serial**
run at `OMP_NUM_THREADS=8`, which should reproduce the parallel output exactly. Not run.

Worth recording regardless: per-expert AUCs carry about +-0.003 of numerical tolerance from thread
count alone. That sits below the effect sizes being resolved (smallest real deltas ~0.03) and the
gates are on medians, but the null-gate tolerance is itself 0.002, so these numbers were never
bit-reproducible across machine configurations.

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

### 3f. Four models flagged by the null battery — reported, not resolved

Under the stop rule, these four fall outside median iid-null AUC 0.500 +- 0.002:

| run | median iid null AUC |
|---|---|
| `g3_moe_s0_1e16_sigmoid_seed2` | 0.5025 |
| `g3_tmoe_s0_1e16_mom` | 0.4975 |
| `flame512_g1_temporal` | 0.4979 |
| `flame38m_g1_moe` | 0.5020 |

**Assessment, which is an interpretation and not a measurement:** the battery samples layer 2 and 24
experts, where a median carries roughly +-0.002 of sampling noise by itself, so these sit about one
standard error from 0.500. The authoritative gate inside `delex_locus_driver` covers every layer and
every expert — 832 to 2496 per model — and passed on all 26 at a maximum deviation of 0.0005.

To settle it rather than argue it: re-run the battery on these four with more layers and experts. Not
done. Their numbers are in use on the strength of the full-depth gate.

### 3g. L3 disagreement between the seed and granularity variations — unresolved

Exempting layer 3 costs +0.068 nats under the granularity variation but +0.039 under the seed
variation, while every other interior layer agrees to within 0.006 across the two. One run per cell
means there is no error bar on a single layer, so this cannot be called a granularity effect or
dismissed as noise. It needs a replicate at the same granularity. The L9 endpoint spike does not
depend on it — that reproduces at nearly the same magnitude in both models and again in the
non-temporal control.

### 3h. T1–T4 — deliberately not started

Out of scope by explicit instruction. Everything they need is in §2.
