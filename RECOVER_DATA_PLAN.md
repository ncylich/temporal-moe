# Pod-deletion recovery plan

**Written 2026-08-24, after the training pod was deleted.** Every claim below was verified
against `origin/layer-lexicality` (tip `95c6ab35`) and the four public Hugging Face repos,
not recalled from session history. The audit method is in the appendix so it can be re-run.

**Scope.** The pod carried `/workspace`, which was gitignored by policy
(`analysis/residency/INSTRUCT_ANALYSIS_PLAN.md` §"Storage policy": *"No large binaries in
git… Token dumps and any future big artifacts: workspace disk + HF hub"*). The HF half of
that policy was never executed for the August adaptation program — all four
`ncylich/temporal-moe-*` repos were last written **2026-07-27**, three weeks before it ran.

**Headline.** Everything measured survived. What died is three things, and one of them
(the d7 prompt pool) is the only item with no committed builder.

---

## 0. Status at a glance

| | item | state |
|---|---|---|
| **Recover** | d7 prompt pool (9,173 prompts) | lost, **no builder committed** — rebuild from prose |
| **Recover** | self-generated CE trajectories (`gemma4_*.pt`, `qwen35_*.pt`) | lost, builder committed |
| **Recover** | gemma4 `d12` adapter | lost, trainer committed |
| **Recover** | qwen35 `d12r2` adapter | lost, trainer + merge script committed |
| Don't recover | half-grain split checkpoints | program closed negative, absent from paper |
| Don't recover | merged serving checkpoints | derived; deliberately deleted pre-loss |
| Don't recover | `instruct-traj/genbench_tokens/` | only consumer is an invalid-by-construction metric |
| Don't recover | `gemma_active_sets.json` | screening instrument; affects re-selection, not any published number |
| Don't recover | crossmodel / frontier 50M-token adapters | not cited by the paper |
| Don't recover | base weights, venvs, `corpus_candidates/` | re-fetchable |
| Intact | all 520 per-item generation dumps (169 MB) | in git |
| Intact | all 18 paper figures' producers and inputs | in git |
| Intact | OLMoE program artifacts, phase0 runs, corpora | on Hugging Face |

**Before executing any of this: check whether the RunPod network volume survived.**
`scripts/residency/snapshot_cells.sh` documents that `/workspace/olmoe-adapt/data` — the exact
directory holding both adapters and the d7 pool — lived on a **network volume**, not pod-local
disk, and that the volume "today truncated two checkpoints and one log mid-write when it hit a
quota that `df` does not report". Deleting a pod does not normally delete an attached network
volume. If it survived, Part 1 is unnecessary. Confirm this first; it is the difference between
an afternoon and a week.

---

# Part 0 — Every re-run the paper wants, in priority order

Added 2026-08-24. This part exists because the recovery work and the paper's remaining
measurements got tangled together, and most of the paper's are **not** gated on recovery.
Group A needs nothing that the pod deletion took. Do it first, and do it independently.
Full statements of each item live in `paper/TODO.md`; this is the index and the ordering.

## Group A — paper-critical, needs only base weights

Nothing here touches a lost adapter, trajectory or prompt pool. All of it can run on a fresh
pod the day it boots.

| # | run | why it matters | cost |
|---|---|---|---|
| A1 | **Qwen3.5-35B, free routing, IFEval, at 16384** | The one arm the truncation sweep missed. Its two constrained arms were rerun and are clean, but with no matched free arm the whole cell falls back to 8192, where the free arm is 8.0% truncated. This is the single number holding Qwen3.5's thinking-on mean at -7.0, so it decides whether "free-form thinking amplifies the constraint" survives for that model. | 1 cell |
| A2 | **WritingBench at 8192 for gpt-oss-120b, gpt-oss-20b, LFM** | Never swept for truncation. At its 4096 budget those three sit at 30-36%, 20-25% and 21-27% on both arms, and Section 6 leans on WritingBench for "prose is the robust surface". The paired delta may survive since both arms truncate alike, but it is untested. | 3 models, 1 surface |
| A3 | **A trained temporally-coherent router, serving measurement** | Every phone number prescribes turnover as a stand-in. LEDGER S3-9 calls this "the required next step, not more systems work". The longest-standing open item in the paper. | new training |
| A4 | **gemma4 thinking-on IFEval and MMLU at double budget** | The last two cells above the 2% cap-hit bar (6.5% and 6.1%), left by judgment rather than evidence. Low value alone; worth folding in only if the machine is already up for A1. | 3 arms x 2 tasks |
| A5 | **Mohsen's Intel iGPU numbers** | Third device class. Not blocking. | external |

**A1 is the highest-value single cell in the whole list.** It is one rerun, it needs no
adapter, and it settles a claim the paper currently has to hedge.

## Group B — paper-critical, gated on Part 1

These need an adapter, so they sit behind the rebuild chain below.

| # | run | why it matters | gated on |
|---|---|---|---|
| B1 | **Adapted gemma4 at 8192 on HumanEval** | Section 7 compares released against adapted at the 1,536-token budget the adaptation runs share, because the adapted model was never run higher, while Section 6 reports the same released cell at 8192. A clause in Section 7 names the budget for now; this retires it. The base side already exists as `gemma4_instruct_cap8k`. | §1.3 |
| B2 | **Qwen3.5 thinking-on adaptation, then its length grid** | Section 7's length result rests on gemma4 alone. Qwen3.5's adapted checkpoints were only ever evaluated thinking-off, where neither routing regime lengthens and the comparison is empty. | §1.1, §1.2, §1.4 |

## Group C — durability, do while the above runs

| # | change | why |
|---|---|---|
| C1 | **Make the dump schema self-describing.** Write `total_toks`, `think_toks`, `answer_toks` as three explicit fields. | Inferring which convention a dump uses produced four separate wrong results on 2026-08-24 (`results/ablations/INSTRUCT_RESULTS.md` §5). Fold into the next regeneration rather than running a pass for it. |
| C2 | **Mirror every adapter and trajectory to Hugging Face as it is written**, and add it to `results/MANIFEST.csv`. | The standing rule from Part 2. It is what would have made this whole plan unnecessary. |

## What is explicitly NOT being re-run

- **GSM8K and IFEval regeneration.** Their cells predate the unfinished-thinking scoring fix,
  but the measured blast radius is 0.73% of GSM8K items and 1.88% of IFEval items, both under
  the binomial standard errors already reported. Re-parsing is impossible (no raw text saved)
  and not worth doing. Recorded so it is not re-litigated.

---

# Part 1 — What we need to recover

The four items form one dependency chain. Each is the input to the next, so they must be done
in order:

```
d7 prompt pool  ->  self-gen trajectories  ->  adapter training  ->  merge + re-measure
   (1.1)                  (1.2)                    (1.3 / 1.4)
```

---

## 1.1 The d7 prompt pool — **the hard one**

### What it was

The 9,173-prompt, benchmark-free corpus that both adapters were trained on. Composition, from
`results/ablations/gemma_adapt_RESULTS.md` §"Recipe (all settings load-bearing)":

| lane | prompts | note |
|---|---|---|
| `domain8k` | 4,958 | includes 431 code rows |
| `mathlane_v2` | 2,341 | |
| `d5` few-shot variants | 1,183 | |
| `mcq-writer` | 691 | |
| **total** | **9,173** | |

Two constraints on it are explicitly load-bearing:

1. **Lineage ban.** *"no benchmark-family data in any form (test/train splits, synthetic
   derivatives)"*. The same doc records why, under §"Why these settings (the ladder)" item 4:
   Orca-Math (seeded from GSM8K-train) produced *"a fake +8 GSM8K that vanished when the lane
   was removed (D1 vs D4 ablation) — style-matching, not constraint robustness."*
2. **8-gram screening** of every lane against the GSM8K, MMLU, HumanEval and IFEval **test**
   sets.

`results/ablations/halfgrain_RESULTS.md:109-111` records an independent audit of the screen
holding: *"mathlane prompts 0/2793 overlaps; generation hits are 1-3 [gram]… Lineage rule
maintained (mathlane_v2 = …)"*.

### Where it was

`/workspace/olmoe-adapt/data/d7_prompts.jsonl` — the path shape is confirmed by
`analysis/residency/selfgen_traj.py:9`, whose usage line reads
`--prompts d7_prompts.jsonl`, and by `analysis/residency/gen_traj_vllm.py:9`, which shows the
sibling `--prompts /workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl`.

### Why this one is hard

**There is no committed builder.** A full-branch grep for `domain8k`, `mathlane`, `mcq-writer`
and `mcq_writer` across every `.py` and `.sh` returns **zero hits**. The three strings appear in
exactly two files on the entire branch, both prose:
`results/ablations/gemma_adapt_RESULTS.md` and `results/ablations/halfgrain_RESULTS.md`.

The committed corpus tooling builds *other* things and is not a substitute:

- `analysis/residency/build_wildchat_prompts.py` — freezes 500 WildChat prompts for the
  self-CE / functional-displacement program. Deterministic, sha256-stamped, *not* the d7 pool.
- `analysis/residency/fetch_corpus_candidates.py`,
  `analysis/residency/score_corpus_candidates.py` — pretraining-corpus selection by bits per
  byte (FineWeb-Edu / DCLM / Nemotron-CC) for the OLMoE and Qwen BPB programs. Different
  problem entirely.
- `analysis/residency/build_qwen_train.py`, `build_qwen_slice.py` — the qwen BPB slice builders.

### Rebuild procedure

1. Reconstruct the four lanes to the counts in the table above. Treat
   `gemma_adapt_RESULTS.md` §Recipe as the specification.
2. Re-apply the 8-gram screen against all four benchmark **test** sets. Enforce the lineage ban
   at lane-selection time, not by post-filtering — the D1-vs-D4 ablation is the evidence that
   post-filtering is insufficient.
3. **Commit the pool and a builder script.** This is the whole point. A pool that exists only as
   a `.jsonl` on a disk reproduces exactly one loss event from now.
4. Record a sha256 next to it, in the style `build_wildchat_prompts.py` already uses (*"plus a
   meta json with the sha256 of the jsonl, so any later regeneration can be checked
   byte-identical"*).

### Cost and caveat

Mostly CPU; hours, not GPU-days. **But this is a rebuild, not a reproduction.** A differently
constituted pool can move the result — that is precisely what the D1/D4 ablation demonstrates.
Combined with the ±2-point single-run screening noise documented in
`gemma_adapt_RESULTS.md` §"Measurement discipline", adapters trained on a rebuilt pool will land
*near*, not *on*, the published Section 8 numbers. See §1.5 for how to handle that.

**Priority note: this is the only irreplaceable-and-cheap item on the list.** Steps 1.2–1.4 are
expensive but perfectly repeatable once 1.1 exists. Do 1.1 before booking a machine.

---

## 1.2 The self-generated CE trajectories

### What they were

Each base model's own responses to the d7 pool, generated **with no residency constraint**,
which the adapters then train on with plain cross-entropy under the constraint. Per
`gemma_adapt_RESULTS.md` §Recipe:

> Trajectories: the model's own think-off responses (sampling per its generation config,
> seed 1234), generated under NO constraint, 2048-token cap, truncation gate <15% capped.

For qwen, `results/ablations/gemma_adapt_RESULTS.md` §"Qwen result: r2" specifies the **clean
pool** variant instead: *"the truncation-free pool (3072-cap regen, rows >2560 dropped whole)"*.
That regeneration is part of the r2 recipe and must be redone, not skipped.

### Where they were

`/workspace/instruct-traj/{tag}.pt`, one file per tag, schema
`{"rows": [{idx, prompt_len, ids}], "meta": {...}}`. Tags referenced in committed code:
`gemma4_train5k` (`train_gemma_ce.py:107`, `:20`), `qwen35_train5k`
(`train_qwen_ce.py:11`, `:28`). The think-on variant is named explicitly in
`gemma_adapt_RESULTS.md:78`: *"Think-on trajectory file for a future resized run:
/workspace/instruct-traj/gemma4_d7think.pt"*.

### Rebuild procedure

The builder **is** committed: `analysis/residency/gen_traj_vllm.py` (bulk vLLM path; its
docstring carries the invocation) with `analysis/residency/gen_trajectories.py` as the
single-stream reference implementation. Usage from `gen_traj_vllm.py:9-10`:

```
gen_traj_vllm.py --model /dev/shm/gemma4-26b-it --tag gemma4_train5k \
    --prompts /workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl
```

Substitute the rebuilt d7 pool for `--prompts`, and set `--think off` for both models. Relevant
flags: `--max-new`, `--max-prompt-tok 512`, `--gpu-mem` (the docstring notes 72GB-class models
need ~0.97). Generation resumes from partial output, so an interrupted run is not lost.

### Cost

Generation-bound. Hours on a large card for each model. No training.

### Durability

Push the `.pt` files to Hugging Face **as soon as they are written**, before training starts.
`docs/research/olmoe-adaptation-plan.md` §"Stage 2 — router-only finetune" already carried this
rule — *"Durability: push checkpoints off-pod at every eval point (the a6000 wipe precedent)"* —
and the August program did not follow it. That omission is the entire reason this document
exists.

---

## 1.3 The gemma4 `d12` adapter

### What it was

The paper's Section 8 gemma result. From `results/ablations/gemma_adapt_RESULTS.md` §Headline —
damage against the *unconstrained* base in percentage points, negative = worse, at the tight
arm R=8=k:

| R8 arm | GSM8K | IFEval | HumanEval | MMLU |
|---|---|---|---|---|
| base (unadapted) | −6.0 | 0.0 | −6.1 | −0.2 |
| **d12 (adapted)** | **0.0** | **−1.0** | **−1.2** | **−1.8** |

### Where it was

`/workspace/olmoe-adapt/data/gemma_ce_d12_adapter.pt`, alongside the full ladder
`gemma_ce_d{5,7,8,9,10,11}_adapter.pt` (`gemma_adapt_RESULTS.md:77`).

### Recipe

From `gemma_adapt_RESULTS.md` §Recipe, all settings marked load-bearing:

- **Surface:** expert-tensor LoRA r16 on the fused 3D expert tensors (grouped-mm path) +
  attention LoRA r32.
- **Constraint ON during CE**: R8, per-row `enforce_from` = prompt length, batched rows.
- **KL anchor weight 0.05** against precomputed base top-50 free-routing logprobs on response
  tokens. CE backward runs before the KL forward (one live graph); KL scoring checkpointed.
- micro-batch 2 / seq 4096 / 16 rows per step; lr 3e-5.
- **Budget 3.4M response tokens.** More is worse — §"Why these settings" item 3 records that at
  10M tokens the KL-0.1 recipe collapses (constrained GSM8K +4 → −10).

The KL weight is a dial, not a default: item 2 of the same section records no-KL (d7) → R8 MMLU
−0.7 but a weak free arm; KL 0.1 (d8) → free arm repaired but R8 MMLU −2.9; KL 0.05 (d12)
interpolates and is the strongest constrained row of the program.

### Rebuild procedure

Trainer: `analysis/residency/train_gemma_ce.py`. Its docstring carries both the invocation and
the design rationale (why hard labels suffice, how the constraint is applied per row under
micro-batching, why the grouped-GEMM expert path exists: 98 → 2900 tok/s).

```
train_gemma_ce.py --traj <gemma d7 traj tag> --tokens 3400000 \
    --expert-lora-r 16 --out .../gemma_ce_d12_adapter.pt
```

Run `--smoke` first. Per the docstring it exercises grouped-path parity against the eager loop,
LoRA engagement/restore, batched-plumbing exactness, free/constrained batch parity, gradient
flow, timed steps, and save/reload.

**Merging is a solved problem — use the fixed code path.** Two merge bugs were found and fixed
at source during the 2026-08-24 overnight run (`TODO.md` §6 Phase 2, commits `ae505b7`,
`d0f67aa`): `--expert-lora-r` must be passed at merge time or unsloth builds attention-only LoRA
modules that silently miss the `elora_gu_A/B` / `elora_dp_A/B` tensors; and
`processor_config.json` must be copied from the source checkpoint into the merged dir or vLLM's
engine boot fails on gemma4's multimodal processor class. Both fixes are committed. A worked
end-to-end invocation is in `analysis/writingbench/wb_matrix3.sh:71-77`.

### Cost

~1 GPU-day at 3.4M response tokens, per the paper's own framing (Section 8: *"One GPU-day of
constraint-aware finetuning"*).

---

## 1.4 The qwen35 `d12r2` adapter

### What it was

The paper's Section 8 qwen result. From `gemma_adapt_RESULTS.md` §"Qwen result: r2 (COMMITTED
2026-08-18)" — all cells against base-free (absolutes: GSM8K 84.5 / IFEval 89.0 / HumanEval
92.1 / MMLU 93.0):

| arm | GSM8K | IFEval | HumanEval | MMLU |
|---|---|---|---|---|
| free | −2.5 | −3.0 | 0.0 | +0.9 |
| R8 | −3.5 | −6.0 | −1.2 | −0.4 |
| R16 | −3.5 | −6.0 | −1.8 | −1.3 |

### Where it was

`/workspace/olmoe-adapt/data/qwen_ce_d12r2_adapter.pt` (`gemma_adapt_RESULTS.md:98`;
`analysis/writingbench/wb_matrix3.sh:99`). Note `train_qwen_ce.py`'s `--out` *default* points at
`/workspace/qwen35-adapt/data/` — the actual artifact lived under `olmoe-adapt/data`.

### Recipe — differs from gemma in four ways

Per §"Qwen3.5-35B-A3B replication" and §"Qwen result: r2":

- **expert-LoRA r8**, not r16 — capacity-matched to gemma's 1.4B.
- **KL 0.1**, not 0.05.
- **Clean pool** (3072-cap regeneration, rows >2560 dropped whole).
- Accommodations forced by 70GB of weights on an 80GB card: paged 8-bit Adam, the **HF stack
  rather than unsloth** (*"unsloth's batched constrained path drifts 4.9% on qwen where plain HF
  shows 0.0–0.3%"*), chunked-checkpointed CE, per-row KL forward, cuDNN SDP off.

r2 was selected by a max-min criterion over a 2×2 attribution square {old/clean pool} × {KL
0.05/0.1}, records `qwen35_ce_d12r{,2,3,4}`. The square's finding is recorded and worth
preserving: effects are **not additive** — KL-0.1's IFEval/MMLU repair only materialises on the
clean pool, while the clean pool's dropped long rows cost 2–4 GSM8K points that only the old
pool recovers.

### Rebuild procedure

Trainer: `analysis/residency/train_qwen_ce.py` (the gemma recipe on the qwen unsloth stack, per
its docstring), which reuses `analysis/residency/train_unsloth.py` machinery.

```
train_qwen_ce.py --traj <qwen clean-pool traj tag> --tokens 3400000
```

**Merge with `analysis/residency/qwen_ce_patch.py`, never with `--merge-out`.** This is the one
piece of hard-won engineering in the chain and its rationale is documented at length in
`TODO.md` §6 (Phase 2 "BLOCKED on a real architectural gap" and Task #78). Summary: the CE
trainer only ever holds Qwen3.5's **text-only submodule** in memory, so `save_pretrained()`
writes a checkpoint that genuinely never contained any `visual.*` tensors, and this vLLM version
has no working text-only serving path for the family — both the self-produced config and a
corrected multimodal config fail, with different errors. `qwen_ce_patch.py` instead streams the
full multimodal base shard-by-shard and patches only the text-side deltas, so the vision tower
survives because it is never dropped. It follows the pattern already proven in
`analysis/residency/qwen_half_split_patch.py`.

The script carries its own closing assertion that every adapter tensor was consumed. That
assertion earned its keep once already (Task #78: 80 of 371 tensors unused, because attention
`q/k/v/o_proj` LoRA sits on every 4th layer and a `sorted(T)[:15]`/`[-15:]` sample never
surfaced layers 10+). Both the missing branch and the check are now in the committed script.
Expect ~371 tensors: 160 expert-LoRA + 80 attention-LoRA + 131 replaced-verbatim.

### Cost

~1 GPU-day training. The merge is a ~72GB, 14-shard streaming copy — disk-bound, no GPU.

---

## 1.5 Re-measurement, and the honesty problem

Once an adapter exists, regenerate its evaluation grid the way the overnight run did:
GSM8K 200 / IFEval 200 / HumanEval full / MMLU-dual, arms free/R8/R16, `--gpu-mem 0.94`, via
`analysis/residency/instruct_genbench_vllm.py` (dumps are now default-on and count-verified).
That produced the committed `gemma4_ce_d12_freshregen` and `qwen35_ce_d12r2_freshregen` records
(commits `d0f67aa`, `d6d8559`), 12 dumps each, all item counts verified.

**The problem.** Because §1.1 is a rebuild, retrained adapters will not reproduce the published
Section 8 table exactly. Pick one of two honest paths and say which in the paper:

- **Re-measure and update Section 8** to the new run, and release those adapters. Clean, but the
  numbers move.
- **Keep the published numbers** (they are fully evidenced — see Part 3) and release the
  retrained adapters **explicitly labelled as replications with their own table**.

Do not ship a retrained adapter under the published numbers. `gemma_adapt_RESULTS.md`
§"Measurement discipline" also warns that screening deltas overstate full-instrument deltas, so
compare like with like: the authoritative 200-item grid, not the screening subsets.

### Open levers, unblocked by recovery

Both are recorded in `gemma_adapt_RESULTS.md` §Open and are currently impossible:

- gemma free-arm MMLU cost (−2.8): untested lever is a KL bracket at 0.03 / 0.07.
- qwen IFEval, still −3.0 free / −6.0 at R8: next lever is a pool regeneration keeping long rows
  **and** zero truncation (cap ≥4k plus the chunked-head trainer, which now has the memory
  headroom).

---

# Part 2 — Execution order

0. **Run Part 0 Group A first, in parallel with everything else.** None of it is gated on
   recovery, A1 is a single cell, and it settles a claim the paper currently hedges. Booking a
   machine for the rebuild is also the moment to run A1, A2 and A4, which share the same base
   weights.
1. **Check the network volume.** If `/workspace/olmoe-adapt/data` survived, stop; nothing below
   is needed.
2. **§1.1 rebuild the d7 pool, commit it and a builder.** CPU. Do this before booking a machine.
3. **§1.2 regenerate trajectories, push to HF immediately.** Both models; qwen needs the clean
   pool variant.
4. **§1.3 gemma d12** → smoke → train → merge via the fixed `train_gemma_ce.py --merge-out`
   path → **push adapter to HF** → re-measure.
5. **§1.4 qwen d12r2** → train → merge via `qwen_ce_patch.py` → **push adapter to HF** →
   re-measure.
6. **§1.5** decide and apply the Section 8 disposition.
7. **Part 0 Group B**, which is unblocked once §1.3 and §1.4 land: B1 needs only gemma, B2
   needs qwen and a thinking-on evaluation pass.

**Standing rule from here on.** Mirror every adapter and trajectory file to Hugging Face the
moment it is written, and add it to `results/MANIFEST.csv` so `scripts/artifacts.py` can fetch
and sha256-verify it. `scripts/residency/snapshot_cells.sh` already argues this case for a 56 KB
artifact — *"There is no reason for the record of a program's results to be less durable than
the code that produced them"* — and the argument holds a fortiori for a 1-GPU-day adapter.

---

# Part 3 — Lost, and not recovering

Each entry states what it was and the specific evidence that dropping it is safe.

### Half-grain split checkpoints

`gemma4-halfgrain-s{1,2}`, `qwen35-halfgrain-s{1,2}` and the adaptation ladder built on them.

**Why it's safe.** The program is closed with a negative answer.
`results/ablations/halfgrain_RESULTS.md` opens: *"Program question: does finer grain make
rolling residency cheaper or better? Answer: no at equal bandwidth (provably a no-op),
partially at half bandwidth after adaptation, with two structural walls that survived every
training-side lever. Program closed."* No `\includegraphics` in the ICLR draft references it and
the string "half-grain" does not appear in `main.tex`. The per-model expert-usage evidence
survives regardless as ten committed `results/ablations/functional_displacement_usage_*.npz`
files, and the splitting code (`analysis/residency/split_experts.py`,
`qwen_half_split_patch.py`) is committed — the checkpoints are regenerable if the question is
ever reopened.

### Merged serving checkpoints

`/dev/shm/gemma4-d12-merged`, `/dev/shm/qwen35-r2-merged`, and the 65–67GB intermediates.

**Why it's safe.** They were derived artifacts, deleted deliberately *before* the pod loss, with
the reasoning recorded at the time (`TODO.md` §6): *"reproducible from
`qwen_ce_d12r2_adapter.pt` + the fixed script in well under an hour if needed again."* They are
a function of the adapters, so recovering §1.3/§1.4 recovers these.

### `/workspace/instruct-traj/genbench_tokens/`

Per-item **token-ID** dumps written as a side output by `instruct_genbench_vllm.py:247`,
`humaneval_think.py:131`, `humaneval_gptoss.py:113`. Distinct from
`results/ablations/genbench_samples/`, which holds the raw text and scores and **is** committed.

**Why it's safe.** Its only analytical consumer is
`analysis/residency/rescore_answer_only.py`, and `results/ablations/DATA_CONTRACT.md` §"Invalid
by construction" lists `metrics *,answer-only` as *"rescores of since-overwritten
generations"* — permanently excluded from the live grid by `partition_eras.py`. Verified
directly: **0 rows** matching `answer-only` in `results/ablations/instruct_genbench_vllm.csv`.
Nothing live depends on it.

### `gemma_active_sets.json`

The screening instrument's active-item subsets (GSM8K 50-of-200, IFEval 70-of-200, derived from
12 runs), referenced by `DATA_CONTRACT.md` §"Screening layer (2026-08-15)" as living in a
scratchpad. Never in git.

**Why it's safe — with a caveat.** `DATA_CONTRACT.md` is explicit that screening is a
**relative** instrument: *"absolute screening scores are NOT comparable to full-run rows… Read
only deltas between records measured under the identical screening protocol"*, and
*"Candidates cited anywhere must first get a full 200-item confirmation grid in the live CSV."*
Every cited result cleared that bar, and `screening_genbench.csv` is committed. **Caveat:** qwen
`r2` was *selected* by the 2×2 screening square (§1.4). Re-running that **selection** — as
opposed to reproducing the known winner — would need regenerated active sets, and they would not
be the same items. The contract already anticipates this: *"regenerate as runs accumulate."*

### Cross-model / frontier 50M-token adapters

`unsloth_*_adapter.pt`, `unsloth_distill100M_T1_lr*_adapter.pt` and similar from the earlier
BPB-based programs (`results/ablations/crossmodel_RESULTS.md`,
`analysis/residency/frontier_qwen.py:28-29`).

**Why it's safe.** Not cited by the ICLR draft. The paper's BPB material is entirely the
from-scratch isoFLOP work of Sections 3–4, whose checkpoints are on Hugging Face
(`ncylich/temporal-moe-ckpts`, 975 files in `results/MANIFEST.csv`). The crossmodel results
themselves remain committed as CSVs and prose.

### Base model weights, virtual environments, corpus candidates

`/workspace/instruct-models/*` (gemma4-26B-IT, Qwen3.5-35B-A3B, OLMoE-0125-Instruct,
LFM2.5-8B-A1B; gpt-oss pulled on demand in `wb_matrix3.sh`), `/workspace/olmoe-adapt/model`,
`olmoe-adapt/venv`, `FLAME-MoE/.venv`, `venv_fla`, `/workspace/corpus_candidates/`.

**Why it's safe.** All public or rebuildable. Environments from `requirements.lock.txt` +
`docs/ENVIRONMENT.md`; corpus candidates from `analysis/residency/fetch_corpus_candidates.py`.

### Superseded pre-fix dumps

398 of the 464 pre-regeneration dump files carry no raw text (they predate the Task 0
dumps-default-on fix). Not recoverable, and not worth recovering: they are either superseded by
a `freshregen` / `cap8k` / `cap16k` sibling, or belong to the item-level evidence gaps
`DATA_CONTRACT.md` already declares *"by construction, not error"* — `mmlu_flan_cot_fewshot`
(an lm_eval group task, no samples), `mmlu_gptoss_relaxed`, and `humaneval_gemma_fixed`, about
35 of 122 live cells, whose CSV rows are the only record and always were.

---

# Part 4 — Verified intact (so nobody re-audits this)

### Every generation from the August regeneration campaign

520 dump files, 169 MB, in `results/ablations/genbench_samples/` on `origin/layer-lexicality`.
All 520 parse; none corrupt. By era:

| era | files | with per-item raw text |
|---|---|---|
| `freshregen` — adapted gemma4 + qwen35, 12 cells each | 24 | 24 / 24 |
| `cap8k` — the 8192 resume sweep (20 HumanEval cells + 3 qwen MMLU arms) | 23 | 23 / 23 |
| `cap16k` — the three IFEval full reruns (tasks #79/#80/#81) | 9 | 9 / 9 |
| pre-fix originals | 464 | 66 |

**All 56 campaign outputs carry full raw per-item text.** Resumed items additionally carry
`resumed_from` and `prefix_source`, so exact-engine-ID continuations remain distinguishable from
retokenized ones. The full narrative is `TODO.md` §§1–6; the plan it executed is
`TRUNCATION_RERUN_PLAN.md`; the write-up is `results/ablations/length_extension_RESULTS.md`.

### The analysis chain rebuilds from the repo alone

`length_extension.py`, `truncation_decomp.py`, `think_analysis.py`, `mmlu_unfinished_rescore.py`,
`plot_length_story.py` and `plot_length_decomp.py` all read only
`results/ablations/genbench_samples/`. No `/workspace` reference among them. Verified by
re-running against a clean checkout:

- `results/ablations/length_extension.csv` (56 cells — the Section 7 backbone) regenerates
  **byte-identically**.
- `truncation_decomp.py` reproduces all 44 committed rows **unchanged**.

### Every paper figure

All 18 `\includegraphics` targets in the ICLR draft map to committed producers. 17 of 18 have
zero pod references. The 18th, `analysis/residency/functional_displacement_figure.py:19`,
hardcodes `ABL = "/workspace/temporal-moe/results/ablations"` — that is the **repo's own**
ablations directory as it sat on the pod, and both CSVs it reads
(`functional_displacement.csv`, `router_wasserstein.csv`) are committed. **One-line fix:** route
it through `analysis/paths.py` like every other producer (see `analysis/residency/README.md`
§"Artifacts and paths" for the convention). Not a data loss.

All figure/analysis input CSVs confirmed present: `functional_displacement.csv`,
`router_wasserstein.csv`, `instruct_selfce.csv`, `instruct_genbench_vllm.csv`,
`screening_genbench.csv`, `think_ablation_summary.csv`, `writingbench/cell_stats.csv`,
`length_extension.csv`, `truncation_decomp.csv`, `mmlu_unfinished_rescore.csv`,
`instruct_mmlu_replicates.csv`.

### On Hugging Face (`results/MANIFEST.csv`, 1,352 files, fetched by `scripts/artifacts.py`)

- `ncylich/temporal-moe-ckpts` (975) — phase0 isoFLOP runs, checkpoints, logs.
- `ncylich/temporal-moe-extras` (248) — ablation CSVs, figures, the OLMoE `merged_ce_model`
  (13.8 GB), `olmoe_adapt/` bake checkpoints, `bpb_slice_ids.pt`, `finetune_ids.pt`.
- `ncylich/temporal-moe-router-adapt` (56) — OLMoE router adapters + bake logs and metadata.
- `ncylich/temporal-moe-corpus` (73) — tokenized DCLM, tok16k, tokenizers.

### Also committed

- OLMoE adaptation scripts archived verbatim in `scripts/adaptation/` (see its `README.md` —
  the archive is the record of what produced the published numbers and is deliberately **not**
  interchangeable with `analysis/residency/residency.py`).
- PLE per-cell JSONs snapshotted into `results/ablations/cells/` by
  `scripts/residency/snapshot_cells.sh`.
- WritingBench responses (58) and scores (58) under `results/ablations/writingbench/`.
- The full program record: `PLE_PLAN.md`, `analysis/residency/README.md`,
  `docs/research/olmoe-adaptation-plan.md`, `results/ablations/README.md`,
  `results/ablations/FINDINGS.md`, `results/ablations/DATA_CONTRACT.md`.

---

# Part 5 — Two unrelated defects found during this audit

Neither is caused by the pod loss; both were surfaced by re-running the producers.

1. **`results/ablations/truncation_decomp.csv` is stale and its producer over-collects.**
   The committed file predates the last night's commits by 9 cells. But re-running is not
   sufficient: `truncation_decomp.py` globs `genbench_samples/*.json`, so it now silently pools
   the 8 **adapted-model** cells into a decomposition that is about **base-model** residency
   damage. That moves the headline — group C (generations that ran long but finished cleanly,
   the only group that speaks to derailment rather than to hitting the token wall) reads 3.2×
   normal wrongness on code (n=234) and 1.8× on knowledge (n=378), against the 3.3× (n=208) /
   1.6× (n=351) on record in commit `82355ff`. Decide whether adapted cells belong in that pool
   and add an explicit filter either way. The ICLR draft does not quote these ratios, so this is
   repo hygiene, not a paper correction.

2. **`functional_displacement_figure.py:19` hardcodes a pod path.** See Part 4. One line.

---

# Appendix — how this audit was done

Repeatable from any clone with the remotes configured:

```bash
git fetch origin layer-lexicality
git fetch overleaf main

# 1. every pod path the committed code depends on
git grep -hoE "/workspace/[A-Za-z0-9_./-]+" origin/layer-lexicality -- '*.py' '*.sh' '*.md' \
  | sed 's|/workspace/||' | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -rn

# 2. does an artifact exist in git at all
git ls-tree -r origin/layer-lexicality --name-only | grep -i <name>
git cat-file -s origin/layer-lexicality:<path>

# 3. is it on Hugging Face
curl -s "https://huggingface.co/api/models?author=ncylich"
curl -s "https://huggingface.co/api/datasets?author=ncylich"
git show origin/layer-lexicality:results/MANIFEST.csv | grep -i <name>

# 4. which paper figures depend on what
git show overleaf/main:main.tex | grep -oE "figures/[a-z0-9_]+\.png" | sort -u
git show origin/layer-lexicality:<producer> | grep -nE "/workspace|instruct-traj"

# 5. prove the analysis chain rebuilds
git worktree add --detach /tmp/ll origin/layer-lexicality
cd /tmp/ll && TMOE_ROOT=$PWD python3 analysis/residency/length_extension.py
git diff --stat results/ablations/length_extension.csv     # expect: empty
```

The HF check in step 3 is the one that settles the adapter question: every
`ncylich/temporal-moe-*` repo reports `lastModified` of **2026-07-27**, which predates the
adaptation program described in Part 1 and confirms none of it was ever mirrored.
