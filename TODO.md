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
