# Re-run orchestration on the rebuilt pod

**Written 2026-08-24, on the machine that replaced the deleted pod.** This is the
execution log for `RECOVER_DATA_PLAN.md`: what the environment actually turned out to be,
which of the plan's assumptions survived contact with it, and what is running where.
The plan says *what* to re-run and *why*; this says *how it is being run here*, and
records the decisions taken along the way so none of them is invisible later.

---

## 0. Volume triage — the plan's step 1, answered

**The network volume did not survive.** `/workspace` is mounted fresh
(`mfs#eur-is-4.runpod.net:9421`) and the three directories the recovery chain needs are
absent: `olmoe-adapt/`, `instruct-traj/`, `instruct-models/`. Part 1 of the plan is live.

Still worth doing once, by hand, because it is the difference between an afternoon and a
week: **check the RunPod console for the old volume as a detached volume.** Deleting a pod
does not normally delete an attached network volume. If it is there and re-attachable,
everything in Part 1 below evaporates.

---

## 1. The machine

| | |
|---|---|
| GPUs | 4x H200, 143 GB each, indexed **0-3** |
| lanes in use | **1, 2, 3 — GPU 0 is deliberately left free** |
| CPU / RAM | 96 cores / 2 TB |
| `/dev/shm` | 469 GB (RAM-backed; models staged here) |
| `/workspace` | MooseFS network volume, effectively unlimited |
| `/` | 2 TB overlay, 1% used |

There is no GPU 4. The lane count is three, not four.

### Environment, as rebuilt

Two venvs, mirroring the old pod's split:

| venv | python | contents |
|---|---|---|
| `/workspace/venv_vllm312` | 3.12 | vLLM 0.27.1, torch 2.13.0, transformers 5.12.1, **lm_eval 0.4.12**, ninja |
| `/workspace/venv_fla` | 3.11 | torch 2.13.0, unsloth 2026.8.4, peft, trl, matplotlib — the training side |

Models staged on `/dev/shm`: `gemma4-26b-it` (49 GB), `qwen35-35b-a3b` (67 GB),
`gpt-oss-20b`, `gpt-oss-120b`, `lfm25-8b-a1b`.

**Two environment facts that are not in any existing doc and will bite a rerun:**

1. **`ninja` must be on `PATH`, not merely installed.** flashinfer JIT-compiles its
   sampling kernels at first use and shells out to `ninja` by name. Invoking the venv
   python by absolute path without `export PATH=/workspace/venv_vllm312/bin:$PATH` fails
   deep inside the sampler with `FileNotFoundError: 'ninja'`, long after the engine boots.

2. **The vendored `lm-evaluation-harness` submodule is the wrong harness for these
   drivers.** It is pinned at `0c8c0d8` (v0.4.5) for the Megatron isoFLOP path, and it
   (a) eagerly imports `megatron_lm.py`, which needs megatron installed, and (b) reads
   `transformers.AutoModelForVision2Seq`, removed in transformers v5. More decisively,
   its `simple_evaluate` has neither `samples=` nor `confirm_run_unsafe_code=`, both of
   which `instruct_genbench_vllm.py` passes — so the residency runs were never produced
   with it. Stock `lm_eval` from PyPI is what this venv needs; 0.4.12 has the full API
   surface and all four task names (`gsm8k_cot_zeroshot`, `ifeval`, `humaneval`,
   `mmlu_flan_cot_fewshot`) resolve.

### The residency stack needed one port, and it is gated

`vllm_glue.install()` patches five architectures eagerly, so one stale attribute took out
every run, not just its own. vLLM 0.27.1 renamed the fused-MoE factory
`FusedMoE` -> `FusedMoEFactory` in `lfm2_moe`. Same signature, same call site: bound under
whichever name is present (commit on this branch). Everything else in the glue survived —
the `GPUModelRunner._update_states` hook, olmoe, `qwen3_next` (also Qwen3.5's MoE block),
gemma4 and gpt_oss all still resolve.

Re-gated with the `smoke_vllm.py` gates on this hardware and version:

| model | gate 1 (R=E byte-identical to free) | gate 2 (R=k engages) |
|---|---|---|
| gemma4-26b-it, E=128, R=8 | PASS, 8/8 concurrent requests | PASS, 8/8 outputs change |
| qwen35-35b-a3b, E=256, R=8 | PASS | PASS, 8/8 outputs change |

**Not yet gated: LFM.** Its factory wrap is the exact piece the port touched, and a silent
no-op there produced arm-identical generations once before. Gate it before A2's LFM cells.

---

## 2. Lane assignment

One cell = one engine boot = one GPU, so lanes are independent processes.
Driver: `scripts/residency/pod_rebuild_lanes.sh {a1,a2,a4}`.

| lane | GPU | item | state |
|---|---|---|---|
| A1 | 1 | Qwen3.5 IFEval @16384, arms free/R8/R32 | running |
| A4 | 2 | gemma4 think-on IFEval 8192->16384, MMLU 4096->8192 | running |
| A2 | 3 | WritingBench @8192, gpt-oss-120b / gpt-oss-20b / LFM | blocked on harness staging + LFM gate |

---

## 3. Decisions taken, with reasons

**A1 runs all three arms in one boot, not just the missing free arm.** The plan costs A1 as
"1 cell". Task #80 reran R8 and R32 at 16384 and recorded them as `qwen35_instruct_cap16k`,
leaving free at 8192 where it is 8.0% truncated. Adding only a free arm would stitch it
onto another boot's constrained arms, which is exactly what the batch-fair
same-boot/same-batch protocol exists to prevent — and `TODO.md` section 4 already records
that constrained-arm generations are not reproducible run-to-run. Task #80's arms took
4358 s and 3729 s, so a matched triple costs about three GPU-hours. Recorded as
`qwen35_instruct_cap16k_b`; the Task #80 rows are untouched, per the house pattern of
keeping the old row and suffixing the new one. `--presence-penalty 1.5` is carried over
because Task #80 used it (the qwen3.5-thinking model-card fallback) and dropping it would
break fidelity with those rows.

**A4's "double budget" is per cell, read off the CSV rather than assumed.** Think-on IFEval
sits at 8192 and doubles to 16384; think-on MMLU sits at 4096 and doubles to 8192. An
earlier draft of the lane script had both at 8192, which would have *halved* the IFEval
cell instead of doubling it.

**`truncation_decomp.py` excludes adapted cells by default** (Part 5 defect 1). Exactly 8
cells were pooling in — `gemma4_ce_d12_freshregen` and `qwen35_ce_d12r2_freshregen` at R8
and R16 on HumanEval and MMLU. Base-only restores MMLU group C to 1.6x (n=351), matching
`82355ff`. HumanEval reads 3.2x (n=224) against 3.3x (n=208) there: same ratio within
rounding, larger n because legitimate cap8k/cap16k base cells landed after that commit.
The refresh added no rows, so the committed CSV was already current for base cells — only
the contamination needed removing.

---

## 4. Open decisions — these are yours, not mine

**The d7 pool's lane sources (section 1.1).** This is the gate for all of Part 1 and the
only item with no committed builder. Confirmed here: a grep across **every commit in the
repository**, not just the current tree, finds zero code hits for `domain8k`, `mathlane`,
`mcq-writer` or `mcq_writer`. The only trace of `d7_prompts` in any blob in history is the
usage line in `selfgen_traj.py`'s docstring. Nothing was deleted; it was never committed.

So the specification is four lane names and four counts (domain8k 4,958 incl. 431 code
rows; mathlane_v2 2,341; d5 few-shot variants 1,183; mcq-writer 691), plus two load-bearing
constraints — the lineage ban and the 8-gram screen against four benchmark test sets.
`halfgrain_RESULTS.md` adds one substantive clue: *"mathlane_v2 = benchmark-free by
construction"*, which reads as generated rather than sampled. **What generates the lanes is
a decision with paper consequences and it has not been made.** See section 5.

Schema constraint, from the consumers rather than from prose: `gen_traj_vllm.py` reads
`p["text"]` unconditionally, so rows must carry a `text` field. `selfgen_traj.py` accepts
`prompt` or `text`. Model the builder on `build_wildchat_prompts.py` — deterministic
stream filter, `{idx, ..., text}` rows, plus a meta json carrying the jsonl's sha256.

**The section 1.5 disposition.** Because 1.1 is a rebuild, retrained adapters will land
*near*, not *on*, the published Section 8 table. Re-measure and update Section 8, or keep
the published numbers and release the retrained adapters as labelled replications. Does not
block execution; decide before anything is released.

---

## 5. Operational risks

**`/workspace` is the same class of storage that truncated files mid-write before.**
`snapshot_cells.sh` records that this volume "today truncated two checkpoints and one log
mid-write when it hit a quota that `df` does not report", and installing a package into
`/workspace/venv_fla` today failed with `Stale file handle (os error 116)` mid-write. A
1-GPU-day adapter should not be written straight to it and left there. Write adapters and
trajectories to local disk or `/dev/shm`, then mirror to Hugging Face immediately — which
is the standing rule from Part 2 anyway, and the omission that caused all of this.

**Nothing is mirrored yet.** Rule C2 — every adapter and trajectory to Hugging Face the
moment it is written, plus a row in `results/MANIFEST.csv` — has no enforcement. It is the
one change that would have made the recovery plan unnecessary.
