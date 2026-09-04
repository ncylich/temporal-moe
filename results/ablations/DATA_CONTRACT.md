# DATA CONTRACT — instruct benchmark results (permanent; not a narrative doc)

Structural facts only. This file survives the narrative-doc rewrites; it contains no
conclusions and cites no results.

## Files
- `instruct_genbench_vllm.csv` — AUTHORITATIVE ONLY: one row per
  (record, arm, task, metric). Maintained by `analysis/residency/partition_eras.py`;
  its exclusion rules are the validity definition.
- `superseded/instruct_genbench_vllm_history.csv` — every superseded/probe/invalid
  row, original order. Includes the greedy-era gemma adaptation trio
  (`gemma4_adapted`/`gemma4_ctrl_sft`/`gemma4_instruct`@640), valid only as its
  internally-paired three-way comparison.
- `think_ablation_summary.csv` — derived; producer `think_analysis.py`.
- `genbench_samples/` — per-item outcomes + lengths for lm_eval-driven non-group
  tasks (GSM8K, IFEval, HumanEval via the main driver).
- `/workspace/instruct-traj/genbench_tokens/` (outside repo) — raw token dumps from
  the main driver and `humaneval_think.py`/`humaneval_gptoss.py`; see MANIFEST.md.
- Item-level evidence gaps (by construction, not error): `mmlu_flan_cot_fewshot`
  (lm_eval group task — no samples), `mmlu_gptoss_relaxed`, and
  `humaneval_gemma_fixed` cells have neither samples nor token dumps; their CSV
  rows are the only record. ~35 of 122 live cells fall in this class.

## Protocol (single-pass; producer instruct_genbench_vllm.py + bespoke scripts)
- One generation pass per request at `--gen-cap` (= `max_gen_toks` column):
  2048 non-thinking / 4096 thinking / 8192 thinking-IFEval and gpt-oss IFEval.
  Cap-finishers are degeneracy-flagged in run logs.
- Sampling: model-card recipe incl. presence_penalty and per-mode temps; seed 1234;
  fallback 0.7/0.95; NEVER greedy.
- Stops: eos-only at the engine; think segment stripped (per-arch marker), then task
  stops applied (`genprotocol.py`).
- Bespoke producers (their budgets are the valid exceptions):
  `humaneval_gemma.py` (1536 off / 3072 on), `humaneval_gptoss.py` (2048 low-med /
  4096 high), `humaneval_think.py` (4096), `mmlu_gptoss.py` (relaxed extraction).

## MBPP standard surface (`mbpp_chat`, 2026-09-04)
- Task `mbpp_chat`, producer `analysis/residency/mbpp_chat.py`, records suffixed
  `_mbpp` (`olmoe_instruct_mbpp`, `lfm25_instruct_mbpp`, `qwen35_instruct_mbpp`,
  `gptoss_20b_mbpp`, `gptoss_120b_mbpp`). All 500 MBPP test problems (`limit` = full).
- Prompt: task text plus the three asserts, one fenced Python block requested; the
  LAST fenced block is executed whole (self-test scaffolds included, same rule as
  the recorded gemma `mbpp_gemma` cells). Thinking stripped per family before the
  fence search; an unclosed thinking span or a cap-out with no fence scores 0.
- Sampling: shipped generation_config, seed 1234, budget 8192 (OLMoE 3328 in its
  4096 window; the CSV `max_gen_toks` column records it). Qwen3.5 runs its card
  non-thinking recipe (0.7 / top_p 0.8 / presence 1.5) through the fast
  presence-penalty processor (`TEMPORAL_FAST_PP=1`, verified equal to native on
  the sub-sample). LFM2.5 has no thinking toggle: it always thinks in-band.
- Per-item dumps `genbench_samples/<record>_<arm>_mbpp_chat.json` hold prompt ids,
  raw generation, executed code, pass, thinking tokens, cap and unfinished flags.
  The `*_mbpp40_*` dumps and `mbpp_subsample.csv` are producer validation only.
- Noise floor at n=500: binomial SE per arm 1.5 to 2.2 points.
- `lfm25_instruct_mbpp_cap16k` free/R4 is the fair-budget twin of `lfm25_instruct_mbpp`
  (11% / 13% at cap at 8192) and supersedes it for citation; 8.6% / 10.2% of its items
  still end in an unclosed thinking span at 16384 (`tmoe_mbpp_cap16k.sh`).
- gemma4 and Qwen3.5 adapted finals keep their recorded MBPP rows (`mbpp_gemma`,
  `mbpp_instruct`); the Qwen base is re-measured under `mbpp_chat` for auditability.

## Fair-budget re-runs (`*_cap16k`, 2026-08-24)
Records suffixed `_cap16k` are the SAME cell re-measured at 16384 because the
original was budget-limited (≥5% of items finishing at the cap). They SUPERSEDE
their unsuffixed twin for that (arm, task); the twin is retained un-edited for
era comparison and must not be cited as the current number. Both carry their own
`max_gen_toks`. Pairs (all IFEval, all `prompt_level_strict_acc`):
`gptoss_20b_high` free/R4, `gptoss_120b_high` free/R4/R16 (all @8192),
`qwen35_instruct` R8/R32 (@8192; its free arm was already unsaturated at 0.5%
and was NOT re-run, so free@8192 is the correct paired baseline for both).
Earlier `_cap8k` records follow the same supersede-not-overwrite convention.

## Noise floor
Cells are single runs, n=200 (MMLU 228, HumanEval 164), seed 1234. Binomial SE per
arm ~1.6-3.5 points; paired damage SE ~1.3-3.4 where per-item dumps allow pairing.
Treat |damage| < 2 SE as noise. `think_ablation_summary.csv` carries SE columns.

## Invalid by construction (enforced by partition_eras.py; never in the live file)
smoke_* probes; lfm25_vllm; lfm25_fullset_audit; adaptation-trio records;
humaneval_instruct for thinking/channel models (primed fence breaks their templates);
mmlu_flan_cot_fewshot for all gpt-oss records (any effort) and LFM (extraction
floor; gpt-oss uses mmlu_gptoss_relaxed instead);
metric exact_match,strict-match (inert under chat protocol);
metrics *,answer-only (rescores of since-overwritten generations).

## Screening layer (2026-08-15)
`screening_genbench.csv` holds RELATIVE screening runs only (active-item doc
subsets via `--samples-json`, small-batch). Free-arm screening matches full runs
per-item (49/50); constrained arms are batch-composition sensitive (resident-set
tie cascades), so absolute screening scores are NOT comparable to full-run rows.
Read only deltas between records measured under the identical screening protocol
(validated: 4/4 cells within 2 pts of known full-grid deltas). Candidates cited
anywhere must first get a full 200-item confirmation grid in the live CSV.
Active sets: scratchpad gemma_active_sets.json (GSM8K 50/200, IFEval 70/200,
from 12 runs); regenerate as runs accumulate.
