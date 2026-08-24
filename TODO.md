# Update: trajectory dump repair + truncation fix + fair-budget resume sweep

**Branch:** `layer-lexicality`. **HEAD at time of writing:** `82355ff`. All work below
is committed and pushed; nothing in this update is uncommitted or local-only.

This entry does not modify any other file — it's a standalone record of what was
done, what broke and got fixed, and an exhaustive map of where every result lives.

---

## 1. What we did, in order

1. **Made per-item trajectory dumps default-on** in every generation driver
   (`instruct_genbench_vllm.py`, `humaneval_gemma.py`, `humaneval_gptoss.py`,
   `humaneval_think.py`, `mmlu_gptoss.py`). Previously dump-writing was ad hoc;
   a prior run had silently produced no dumps for a large slice of the grid.
   Drivers now fail loudly at startup if the dump directory isn't writable
   (`genprotocol.check_dump_dir()`), and every dump write is followed by a
   read-back count verification (`genprotocol.write_dump()`), so a dump that
   silently kept 4 of 228 items (a real historical bug) is now impossible to
   ship undetected.
2. **Recovered lengths from existing dumps at zero GPU cost** where the raw
   text already existed (think-off MMLU dumps, WritingBench responses/scores
   rescued from pod-local storage).
3. **Regenerated every length-blind cell with dumps** (Task 3): all 6 models,
   HumanEval + MMLU, every effort level and thinking mode. 44 of 47
   regenerated cells reproduced their original grid rows within 2 SE.
4. **Investigated and explained, rather than blindly re-ran, every
   >2-SE discrepancy** (this mattered — two of three were artifacts of
   *our own tooling*, not the model):
   - gpt-oss MMLU cells moved because the old harness hardcoded ancestral
     `temperature=1.0, top_p=1.0` sampling; the rewritten harness uses the
     model's own recipe, and gpt-oss ships none, so the documented no-recipe
     fallback (0.7/0.95) applies. Verified directly against gpt-oss's
     upstream `generation_config.json` and README (neither specifies a
     recipe) — the new rows are correct, no rerun needed.
   - A qwen MMLU "flag" was withdrawn: it came from a bug in our own
     `mmlu_flan_rescore.py`, which used a filter that grabs the *first*
     "answer is" in the text — but qwen's thinking traces frequently mention
     "answer is" mid-reasoning before the real answer, so the rescore was
     reading deliberation. The harness's own metric was correct all along.
   - qwen HumanEval genuinely moved and was adjudicated by rerunning at the
     protocol-default GPU memory fraction: the new value reproduces itself
     across two independent runs (identical score, but note — see the
     "trajectories aren't reproducible" finding below) and is far above the
     old row, which cannot be independently checked because that old run
     saved no trajectories.
5. **Found and fixed a real scoring bug: unfinished thinking was being fed to
   answer extractors.** When a generation exhausts its token budget *inside*
   its thinking block, there is no closing marker, so the old code's
   "strip everything up to the marker" logic was a no-op — the raw
   in-progress reasoning got handed to the answer extractor. Reasoning text
   is full of phrases like "if X then the answer is (D), otherwise (B)",
   so the extractor was frequently grabbing a mid-thought guess and scoring
   it as the model's answer. Fixed in `genprotocol.py`, `mmlu_gptoss.py`,
   `humaneval_think.py`, `humaneval_gemma.py`: an unfinished response now
   scores as "no answer" instead of extracting from its scratch work.
6. **Diagnosed that truncation was the dominant driver of "residency damage"
   on several headline cells**, per `TRUNCATION_RERUN_PLAN.md`. Built a
   decomposition (`truncation_decomp.py`) splitting every long/blown-up
   generation by how it actually ended: (A) hit the cap with thinking still
   open — emitted nothing; (B) hit the cap after thinking closed, answer cut
   off; (C) ran long but finished cleanly — the only group that's actually
   about derailment; (D) blown only because the *free* arm hit its cap.
7. **Built a resume mechanism instead of full reruns** (`resume_truncated.py`,
   with `--dry-run` for zero-GPU plumbing validation, and
   `rescore_resumed_dumps.py` for offline repair). For every cell where a
   generation hit its original budget, we doubled the budget to 8192 and
   *continued only the truncated generations from their saved prefix* —
   never re-drawing the ones that already finished (avoiding a resampling
   bias) and never re-running a whole cell from scratch. Total cost: 264
   generations continued, ~1.8h of GPU generation time, versus an estimated
   ~35M tokens (many hours) for full reruns of the same 20 cells.
8. **Two bugs surfaced during the resume sweep and were fixed immediately**:
   - `pkill -f` self-matching its own wrapper process when launched via
     heredoc (killed a chain script before it could hand off) — fixed by
     writing scripts to disk and killing by captured PID instead of pattern.
   - **The important one**: `resume_truncated.py` wrote its merged dump
     *before* scoring it for a period, so every `*_cap8k_*` dump written in
     that window held the **stale** per-item `pass` value inherited from the
     truncated original, while the CSV row it also wrote held the *correct*
     re-scored value. Per-item `pass` is what every flip/wrongness analysis
     reads — so the dumps were wrong exactly where it mattered most. Caught
     by cross-checking a fair-budget damage table against the resume logs
     (every row showed zero shift, which contradicted what the logs said).
     Fixed (scoring now runs before the write) and **repaired the 20 already-
     written dumps offline** — no regeneration needed, since scoring is pure
     CPU over already-saved text. The repair was validated by confirming it
     reproduces every cell's original resume-log pass@1 exactly.

---

## 2. Headline finding: how much of "residency damage" was actually a token-budget artifact

Every cell below was re-measured at double its original generation budget
(new cap 8192), with the truncated-generation-continuation method above —
same items, same seed, same everything else, only the budget differs.
Single run per cell (not multi-seed) unless noted.

| cell | arm | old cap | damage @ old cap | damage @ 8192 |
|---|---|---|---|---|
| gemma think-on HumanEval | R8 | 3072 | **−12.2** | **−3.7** |
| gemma think-on HumanEval | R16 | 3072 | −2.4 | +0.0 |
| gemma think-off HumanEval | R8 | 1536 | −5.5 | −4.3 |
| gemma think-off HumanEval | R16 | 1536 | +0.0 | −0.6 |
| qwen think-on HumanEval | R8 | 4096 | −6.1 | −6.1 (unchanged) |
| qwen think-on HumanEval | R32 | 4096 | +1.2 | +0.0 |
| LFM HumanEval | R4 | 4096 | **−15.9** | −15.2 (unchanged — real damage) |
| gpt-oss-20b high HumanEval | R4 | 4096 | −5.5 | −1.2 |
| gpt-oss-20b med HumanEval | R4 | 2048 | +1.8 | −1.2 |
| gpt-oss-120b high HumanEval | R4 | 4096 | −2.4 | −0.6 |
| gpt-oss-120b high HumanEval | R16 | 4096 | +0.6 | −0.6 |
| gpt-oss-120b med HumanEval | R4 | 2048 | +1.8 | +3.0 |
| gpt-oss-120b med HumanEval | R16 | 2048 | −0.6 | +1.2 |
| qwen think-on MMLU | R8 | 4096 | **−6.6** | **−2.2** |
| qwen think-on MMLU | R32 | 4096 | −0.4 | −1.3 |

(damage = 100 × (constrained accuracy − free accuracy); negative = residency
hurts. "Damage" column units are percentage points.)

**Key takeaways:**
- The two biggest reported damage numbers in the whole grid (gemma think-on
  HumanEval R8 at −12.2, qwen think-on MMLU R8 at −6.6) both shrink to
  roughly a third of their reported size once the token budget stops being
  the bottleneck. Most of what looked like "residency breaks the model" was
  "the model ran out of room to finish thinking."
- Not every cell moves — LFM's −15.9 barely changes (−15.2), so that's real
  residency damage, not a budget artifact. This is the discriminating power
  of the sweep: it doesn't uniformly deflate everything, it separates real
  damage from measurement artifact per cell.
- The old claim that "thinking-on roughly triples gemma's residency damage"
  (−12.2 vs −5.5, a claim `TRUNCATION_RERUN_PLAN.md` §8 flagged as resting on
  contaminated data) does **not** survive matched budgets: at 8192 both
  modes are statistically indistinguishable (−3.7 on vs −4.3 off).
- The A/B/C/D decomposition (post-repair, final numbers):
  - **Group A** (hit cap, thinking never closed): HumanEval 173 items,
    97.1% wrong. **Group B** (hit cap, answer cut off): 44 items, 88.6%
    wrong. Both are mechanical — generations that emitted nothing score as
    wrong because there's nothing to grade.
  - **Group C** (ran >2x the free counterpart's length but finished cleanly —
    the only group that's genuinely about derailment/length-quality
    relationship, not budget): HumanEval 208 items at 14.9% wrong = **3.3x**
    the 4.6% wrongness of normal-length generations. MMLU: 351 items at
    22.8% wrong = **1.6x** normal (13.8%).
  - This 3.3x/1.6x signal is the one that should be cited as "long
    generations really are worse," separated from the far larger and
    mechanical A/B truncation effect.

---

## 3. Where every result lives (exhaustive)

**Nothing described below was moved, renamed, or overwritten relative to
prior work** — original-budget cells and rows are untouched; the 8192 results
are additive, living in parallel files/records so both budgets stay
comparable.

### Per-item dumps (raw generations, one JSON per cell)
- `results/ablations/genbench_samples/` — every regenerated Task-3 cell
  (original budgets: 1536/2048/3072/4096 depending on model/effort/thinking
  mode), one file per (record, arm, task).
- `results/ablations/genbench_samples/*_cap8k_*.json` — **23 files**, the
  fair-budget (8192) resumed results: 20 HumanEval cells (gemma think-on ×
  {R8,R16,free}, gemma think-off × {R8,R16,free}, qwen think-on × {R8,R32},
  LFM × {R4,free}, gpt-oss-20b high × {R4,free}, gpt-oss-20b med × {R4,free},
  gpt-oss-120b high × {R4,R16,free}, gpt-oss-120b med × {R4,R16,free}) plus 3
  qwen think-on MMLU arms ({R8,R32,free}). Every item that was resumed carries
  `resumed_from` (its original token count), `prefix_source` (how its
  continuation prefix was derived), and corrected `pass`/`unfinished` fields
  (post-repair).

### Score-level CSV rows
- `results/ablations/instruct_genbench_vllm.csv` — the authoritative grid,
  original budgets, promoted after `partition_eras` moved 89 superseded rows
  to `results/ablations/superseded/instruct_genbench_vllm_history.csv`
  (zero rows lost — every superseded row is still there, just marked
  historical).
- `results/ablations/screening_genbench.csv` — the 8192 fair-budget rows,
  record names suffixed `_cap8k` (e.g. `gemma4_think_on_cap8k`), **26 rows**
  (20 HumanEval cells + 3 MMLU arms × 2 metrics each: relaxed-extract and
  strict-flan). Deliberately kept in the screening CSV rather than merged
  into the authoritative one, since the budget is the variable under test —
  both budgets need to stay independently visible and comparable.
- `results/ablations/instruct_mmlu_replicates.csv` — seed-varied replicate
  runs of qwen MMLU (used to distinguish signal from sampling noise before
  we determined the flag was our own rescore bug, not the model).

### Aggregated / derived analysis (all recomputed after the stale-pass repair)
- `results/ablations/truncation_decomp.csv` — the A/B/C/D decomposition
  above, per cell, **44 cells** total (24 original-budget + fair-budget
  cells combined).
- `results/ablations/length_extension.csv` — flip counts, blow-up shares,
  wrongness-conditional-on-blowup, **56 cells**, tagged `HumanEval`,
  `HumanEval-8k`, `MMLU`, `MMLU-8k`, `WritingBench` by surface/budget.
- `results/ablations/mmlu_unfinished_rescore.csv` — MMLU cells re-scored
  under the "unfinished thinking = no answer" fix, showing before/after per
  cell.
- `results/ablations/prefix_roundtrip_allowlist.json` — records which
  (record, task) pairs have verified-exact token-ID prefixes available for
  resume (vs. approximate/retokenized prefixes) — see caveat below.
- `results/ablations/figures/length_extension_decomp.png` (+ `_nocaption`
  variant) — regenerated from the post-repair `length_extension.csv`.

### Narrative writeup (partially updated — see Outstanding below)
- `results/ablations/length_extension_RESULTS.md` — currently contains the
  qwen-MMLU fair-budget section only (`## Measured at a fair budget: two
  thirds of qwen's MMLU damage was the budget`). **Does not yet include**
  the gemma think-on/off, gpt-oss, LFM, or qwen-HumanEval fair-budget
  results, or the final post-repair A/B/C/D numbers above — those exist in
  the CSVs but haven't been written into prose yet.

### Producers (code, all committed)
- `analysis/residency/genprotocol.py` — default-on dump writing +
  count-verification + unfinished-thinking fix.
- `analysis/residency/truncation_decomp.py` — the A/B/C/D decomposition.
- `analysis/residency/length_extension.py` — flip/blowup analysis, now reads
  both original- and fair-budget cell sets.
- `analysis/residency/mmlu_unfinished_rescore.py`, `mmlu_flan_rescore.py`.
- `analysis/residency/resume_truncated.py` — the resume mechanism
  (`--dry-run` for no-GPU validation, `--old-cap`/`--new-cap`, prefix
  reconstruction, in-run scoring).
- `analysis/residency/rescore_resumed_dumps.py` — the offline repair tool
  for the stale-pass bug.
- `analysis/residency/tests/test_prefix_roundtrip.py`,
  `test_tokenizer_families.py`, `temporal/tests/test_resume_residency.py` —
  correctness tests for the resume/continuation machinery.

---

## 4. Known caveats on the fair-budget (8192) numbers

- **Single run per cell.** None of the 8192 numbers are multi-seed averaged;
  small movements (≤2 points on 164-item HumanEval cells) are within noise.
  The large movements (gemma think-on, qwen MMLU, gpt-oss-20b high) are well
  outside plausible single-run noise.
- **Prefix reconstruction is approximate for resumed generations.** The
  original dumps didn't save exact token IDs for the truncated generations
  (only decoded text), so resuming required re-tokenizing the saved text to
  reconstruct a token-ID prefix. This is not guaranteed to byte-match the
  model's original tokenization in all cases — `test_prefix_roundtrip.py`
  and `test_tokenizer_families.py` characterize where this holds exactly vs.
  approximately per model family; `prefix_roundtrip_allowlist.json` records
  which (record, task) pairs are exact. Newly-generated dumps (going
  forward) save exact token IDs and don't have this issue.
- **Constrained-arm trajectories are not reproducible run-to-run**, even
  though the aggregate score is. Confirmed directly: two independent runs of
  qwen HumanEval R8 at the protocol-default GPU memory fraction produced the
  *identical* pass@1 score but *zero* of 164 generations were textually
  identical between the runs. This means per-item analysis (e.g. "which
  specific item flipped") is only valid within a single run, not across
  reruns of the same cell.
- **gpt-oss-20b high effort still truncates ~6-7% of items even at 8192**
  (10-11 of 164). If this cell needs to be fully truncation-free, it would
  need a further budget increase; not done here.

---

## 5. Outstanding / not yet done

- `results/ablations/think_ablation_summary.csv` is **stale** — it predates
  the resume sweep entirely (last written 22:22 Aug 23, sweep started 01:20
  Aug 24). Needs regenerating via `analysis/residency/think_analysis.py`.
- `length_extension_RESULTS.md` needs the remaining fair-budget sections
  written in (gemma on/off, gpt-oss, LFM, qwen HumanEval, final A/B/C/D
  numbers) — data is ready in the CSVs, just not yet prose.
- Two `--no-caption` figure variants not yet regenerated:
  `think_damage_nocaption.png`, and re-verify
  `length_extension_decomp_nocaption.png` is current post-repair.
- A handful of small uncommitted diagnostic files exist locally (an A/B test
  of a decode-step optimization, unrelated to this sweep — see task #72 in
  the task tracker) — not part of this update, left as-is per "don't move
  anything around."
- Task #73 (gpt-oss sampling recipe question) is resolved in substance (see
  §1.4 above) but still shows as pending in the task tracker — bookkeeping
  only, no action needed on the actual result.

---

## 6. Overnight orchestration (2026-08-24, appended — does not modify sections 1-5)

Launched an unattended overnight chain per Noah's instruction: hourly cron
heartbeat (independent of the Monitor/bash-chain mechanism), persistent Monitor
on stage transitions, commit-and-push after each phase.

**Phase 1 — closed the last truncation gap, SUCCEEDED.** `gpt-oss-20b-high`
(free/R4) was still 6.1%/6.7% truncated at the 8192 cap. Resumed (not
re-run) from the existing 8192 prefixes to 16384; all 21 continued items used
**exact engine token IDs**, confirming the ID-persistence fix carries through a
second-order resume correctly. Truncation now 3.0-3.7%. Committed `7abb259`.

**Phase 2 — adapted-model regeneration (gemma4_ce_d12, qwen35_ce_d12r2).**
Both are LoRA/expert-LoRA adapters that needed merging onto their base models;
no merged checkpoint survived anywhere on disk (checked three ways per Noah's
prompt to look at command history — adapters exist, merges don't).

- **gemma4_ce_d12: SUCCEEDED, committed `d0f67aa`.** Two real merge bugs found
  and fixed at the source in `train_gemma_ce.py` (not worked around):
  1. `--expert-lora-r` was omitted, so unsloth built attention-only LoRA
     modules while the checkpoint actually carries grouped-mm expert-LoRA
     tensors (`elora_gu_A/B`, `elora_dp_A/B`). Confirmed the correct rank (16)
     by reading it directly from the checkpoint's own stored metadata rather
     than guessing.
  2. The merge's `save_pretrained` does not carry gemma4's multimodal
     `processor_config.json`, which vLLM needs even for text-only serving —
     this exact bug was already in memory from an earlier program and simply
     wasn't applied here. Fixed with a copy step after `save_pretrained`.
  A near-miss: the orchestration script's own cleanup (`rm -rf`) would have
  deleted the successfully-merged checkpoint seconds after the *second* bug
  surfaced, forcing a full re-merge for nothing. Caught it by checking
  `/dev/shm` state immediately after the failure notification and killing the
  exact PIDs (not a `-f` pattern) before the cleanup line executed.
  12 dumps, 12/12 with raw text, full CSV coverage (GSM8K/IFEval/HumanEval/
  MMLU-dual × free/R8/R16).

- **qwen35_ce_d12r2: BLOCKED on a real architectural gap, not a quick fix.**
  The `--expert-lora-r 8` fix (confirmed from checkpoint metadata, same
  pattern as gemma) resolved the merge itself cleanly. But the checkpoint that
  results is **unservable by vLLM as of this vLLM version**, and it is not a
  config or CLI-flag problem:
  - Qwen3.5-35B is multimodal (ships `preprocessor_config.json` +
    `video_preprocessor_config.json`, different names than gemma4's file —
    broadened the processor-copy fix to cover all three filenames per family,
    committed `1da222c`, checked proactively before it could fail a second
    time under a different name).
  - The CE-adaptation training pipeline (`--no-unsloth`/HF+peft path) only
    ever loads the **text-only submodule** of the multimodal checkpoint, so
    `merge_and_unload()` + `save_pretrained()` produces a checkpoint whose
    safetensors files genuinely never contained any `visual.*` vision-tower
    tensors — they were never in memory to save. Its self-produced
    `config.json` correctly reflects this (`model_type: qwen3_5_moe_text`,
    `architectures: [Qwen3_5MoeForCausalLM]`).
  - Tested **both** the checkpoint's own self-consistent config and a
    "corrected" config copied from the full multimodal base, with a fresh
    remerge in between so each was a clean test, not a guess stacked on a
    guess. **Both fail**, with different errors:
    - Self-produced (text-only) config: `TypeError: Invalid type of
      HuggingFace config. Expected type: Qwen3_5MoeConfig, but found type:
      Qwen3_5MoeTextConfig` — this vLLM version's Qwen3.5-MoE loading path
      unconditionally calls `get_hf_config(Qwen3_5MoeConfig)` regardless of
      which model class def gets dispatched to, so a text-only config is
      rejected outright.
    - Base's full multimodal config forced onto the merged tensors:
      `ValueError: Following weights were not initialized from checkpoint:
      {113 × visual.blocks.N.norm*, visual.patch_embed.*, visual.merger.*,
      visual.pos_embed.*}` — config now correctly matches a class that
      requires vision-tower weights, but those tensors were never produced by
      training in the first place.
  - **Root cause, plainly:** the merge writes only what the trainer's live
    Python object holds in memory (a text-only submodule), and there is no
    text-only serving path in the installed vLLM version for this model
    family. A `save_pretrained`-style merge cannot fix this.
  - **The correct fix** (not attempted tonight — real engineering, not a
    config tweak): a **patch-onto-a-full-base-copy merge**, the same pattern
    this codebase already uses successfully in
    `analysis/residency/qwen_half_split_patch.py` for the half-grain program —
    copy the complete base checkpoint (vision tower included, untouched by
    this training recipe) and patch in only the delta-changed text-side
    tensors from the adapter, rather than doing a raw `save_pretrained` from a
    partial in-memory model. This is a real script to write, not a flag.
  - Nothing was lost: the adapter (`qwen_ce_d12r2_adapter.pt`) and base weights
    are both safely persisted; the broken 65GB `/dev/shm` output was deleted
    (freed back to 0 used) since it is cheaply reproducible from the adapter
    once the patch-based merge exists. No GPU time was spent past the two
    quick diagnostic probes (1-item each) that established this diagnosis.

**Phase 3 (truncation check on the adapted regen):** only gemma's cells exist
to check (qwen never produced valid dumps). None of gemma's 12 cells hit 5%
truncation at the caps used (2048 GSM8K/IFEval, 4096 MMLU, 1536 HumanEval) —
no resume needed there.

**What's next for qwen adaptation, if picked up later:** write the
patch-onto-base-copy merge script (adapt `qwen_half_split_patch.py`'s pattern
rather than starting from scratch), verify it produces a checkpoint whose
`model.safetensors.index.json` key set is a strict superset of the base's
(vision tower present, text-side tensors patched), then rerun Phase 2/3 for
qwen35_ce_d12r2 only — gemma does not need to be touched again.

**Phase 4 — decode-step CUDA-graph speedup, cross-architecture validation
(task #72).** Previously validated on OLMoE only (bit-exact, 18% real
end-to-end win). Same protocol repeated on a second, structurally different
architecture — qwen3.5-35B (256 experts / 40 layers vs OLMoE's 64/16),
R8 GSM8K=100, think off:

| arm | wall time | exact_match | text vs baseline |
|---|---|---|---|
| baseline (dict walker + eager step) | 234s | 0.7600 | — |
| **fast (slots walker + graph step)** | **202s** | **0.7600** | **100/100 bit-identical** |
| baseline again (run-to-run floor) | 229s | 0.7600 | 100/100 bit-identical |

Bit-exact again, real ~13% wall-clock win (smaller than OLMoE's 18%, consistent
with the already-diagnosed pattern: walker overhead is roughly fixed per step,
so it's a smaller fraction of a bigger model's per-step compute). One
operational note: the first attempt at this failed instantly (`ValueError: No
available memory for the cache blocks`) because the quick A/B script omitted
`--gpu-mem` — qwen35 is a ~70GB-class model that needs the elevated values
(0.92-0.95) used everywhere else tonight; the harness's 0.85 default isn't
enough headroom. Not a walker/decode bug; fixed by adding the flag and
rerunning (no real generation time was wasted, the failure was at engine boot).

Two architectures now confirmed clean. A third (gpt-oss, the MXFP4-quantized
family) is queued next before deciding whether to flip either default.

**Third architecture (gpt-oss-20b, MXFP4) — also bit-exact, and the decision:**
Same protocol, R4 GSM8K=100, think off:

| arm | wall time | exact_match (flexible) | text vs baseline |
|---|---|---|---|
| baseline (dict walker + eager step) | 98s | 0.8800 | — |
| **fast (slots walker + graph step)** | **82s** | **0.8800** | **100/100 bit-identical** |
| baseline again (run-to-run floor) | 104s | 0.8800 | 100/100 bit-identical |

Three architecturally distinct models (OLMoE 64e/16L, qwen3.5-35B 256e/40L,
gpt-oss-20b MXFP4-quantized) all bit-exact, all a real wall-clock win
(-13% to -18%, gpt-oss ≈ -16-21% depending on which baseline run is the
comparator). No case found where the fast path diverges even by one token.

**Decision: flipped the project-wide defaults.** `TEMPORAL_DECODE` now
defaults to `graph` (was `eager`) and `TEMPORAL_WALKER` now defaults to
`slots` (was `dict`), in `temporal/temporal_router.py` and
`analysis/residency/vllm_residency.py` respectively. `TEMPORAL_DECODE=eager`
/ `TEMPORAL_WALKER=dict` still work as explicit opt-outs — every grid row
produced before 2026-08-24 was generated on that eager/dict path, so use the
env vars if a future comparison needs the old path specifically. Full
regression suite (`test_decode_accel.py`, `test_walker_slots.py`,
`test_decode_state.py`, `test_vllm_walker.py` under all 4
TEMPORAL_WALKER×TEMPORAL_DECODE combos, `temporal/tests/test_resume_residency.py`)
re-run clean after the flip. Committed `3bb4a85` (code + the qwen/gpt-oss/OLMoE
validation dumps and CSV rows above as evidence). Task #72 done.

**Task #78 — qwen35_ce_d12r2, the patch-onto-base-copy merge that was flagged
above as "not attempted tonight." SUCCEEDED.** New script:
`analysis/residency/qwen_ce_patch.py`. Same shape as
`qwen_half_split_patch.py`: reads the full multimodal base checkpoint
shard-by-shard (never loads a model into memory), adds the adapter's
expert-LoRA deltas onto `mlp.experts.{gate_up,down}_proj` for the text-side
layers, replaces a handful of full-precision tensors the adapter also carries
(per-layer norms), and copies every other tensor — the entire vision tower
included — through byte-for-byte unchanged. Because the vision tower is never
absent from memory in the first place (unlike `train_gemma_ce.py --merge-out`,
which only ever holds the text-only submodule), the resulting checkpoint's own
config is the untouched multimodal one and satisfies this vLLM version's
Qwen3.5-MoE loader without any config surgery.

One real bug caught before it reached vLLM: the first full run finished all 14
shards and then failed its own closing assertion (`adapter not fully
consumed`) — 80 of the adapter's 371 tensors were unused. Root cause: the
adapter carries attention `q/k/v/o_proj` LoRA (r=32, `lora_alpha=64` → scale
2.0, PEFT's standard `B @ A` layout) on every 4th transformer layer
(3, 7, 11, ..., 39), and the script's key-naming check during development had
only sampled `sorted(T)[:15]`/`[-15:]`, which alphabetically lands on layers
0–9's expert-LoRA and norm keys and never surfaces layer 10+'s attention keys.
The assertion did its job — this was caught immediately after the write, not
discovered later against a bad checkpoint. Fixed two ways: (1) added the
missing `q/k/v/o_proj` LoRA branch to `qwen_ce_patch.py` itself, so future runs
of this script don't hit it; (2) rather than repeat the full 72GB, 14-shard
copy, wrote a one-off script that located the exact 2 shards
(`00013`, `00014` of 14) holding those 10 layers' attention weights via the
index's `weight_map`, and patched only those in place. Re-verified afterward
that all 371 adapter tensors are accounted for across the two passes (160
expert-LoRA + 80 attention-LoRA + 131 replaced-verbatim = 371).

Regenerated end-to-end with dumps at the same recipe as gemma4_ce_d12
(GSM8K=200/IFEval=200/HumanEval=full/MMLU-dual, free/R8/R16, `--gpu-mem 0.94`).
12/12 dumps, all item counts verified (200 GSM8K, 200 IFEval, 164 HumanEval,
228 MMLU per arm — GSM8K's dump has 400 raw entries because lm_eval scores it
under two filters, strict-match and flexible-extract, both logged per item;
the CSV's own `sample_len` column correctly reports 200). No cell hit the 5%
truncation threshold — no resume needed. Committed `d6d8559`. The freed
`/dev/shm` checkpoint (67GB) was deleted after commit, same convention as
gemma4_ce_d12; reproducible from `qwen_ce_d12r2_adapter.pt` + the fixed script
in well under an hour if needed again. Task #78 done — this was the last
open item from tonight's queue.
