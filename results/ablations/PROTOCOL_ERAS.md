# Protocol eras of instruct_genbench_vllm.csv (READ BEFORE USING ANY ROW)

The CSV is append-only history. **Rule: for any (model, arm, task) cell, the LAST row
in the file is authoritative.** A `PROTOCOL CUTOVER 2026-08-13 01:29 UTC` comment
marks where the final protocol begins; the final-rerun chain re-ran every materially
affected cell below it. Rows above survive only where listed under "still valid".

| era | window (UTC) | defects | status |
|---|---|---|---|
| E1 greedy | ... -> 08-12 03:45 | greedy decoding (thinking models degenerate); gpt-oss scored on tag-stripped text | fully superseded |
| E2 sampled grid | 08-12 05:24 -> 14:40 | thinking text judged against task formats; ifeval task-yaml silently capped ALL generation at 1280; qwen missing presence_penalty + mode recipes; humaneval_instruct stop-strings ("\ndef"...) fired inside thinking (LFM) | superseded except "still valid" list |
| E3 ablation + partial fixes | 08-12 14:40 -> 08-13 01:00 | same ifeval/budget defects on lm_eval cells; answer-only ",answer-only" rescores partially corrected scoring only | superseded except "still valid" list |
| E4 FINAL | 08-13 01:29 -> | card sampling recipes (incl. presence_penalty, per-mode temp/top_p); lm_eval native reasoning path (eos-only stops, thinking stripped pre-scoring); CLI budget overrides task-yaml caps; thinking caps 4096 (ifeval 8192); raw token dumps | authoritative |
| E5 fallback fix | 08-13 19:16 -> | E4 plus: no-recipe sampling fallback corrected to community-standard 0.7/0.95 (was HF-ancestral 1.0/1.0, depressed OLMoE 5-14 pts); OLMoE fully re-run | authoritative (supersedes E4 OLMoE rows) |

## Cells above the cutover that remain valid (no E4 rerun needed)
- OLMoE: E5 rows only (last rows in file). Earlier OLMoE rows -- including early-E4 --
  ran under the ancestral-sampling fallback and sit 5-14 pts low; superseded.
- gemma think-OFF: all cells (no thinking text; budgets never bound). Channel-aware
  HumanEval (`humaneval_gemma_fixed`) all arms/modes at 1536/3072 budgets.
- gemma think-ON GSM8K (extraction robust; rescore matched exactly) and HumanEval@3072.
- gpt-oss channel-native HumanEval (`humaneval_gptoss`): all arms/efforts (bespoke
  llm.chat path -- never touched lm_eval stops/caps); high effort valid only at the
  4096 rows (2048 rows purged).
- gpt-oss relaxed MMLU (`mmlu_gptoss_relaxed`): all arms/efforts (until ["</s>"]
  benign, no yaml cap).
- LFM GSM8K (extraction robust, matches model card) and MMLU (extraction-floor
  censored, as always).
- gpt-oss GSM8K at LOW and MEDIUM effort (finals present, analyses well under
  budget). HIGH-effort gsm8k rows above the cutover are INVALID: the original
  "zero cap-outs" audit measured post-filter (final-channel) lengths -- raw-read
  showed 35% empty finals from analyses hitting 2048. High-effort gsm8k re-run at
  cap 4096 below the cutover.

## Known-invalid classes (never cite)
- Any greedy-era generative row (E1).
- Any thinking-model IFEval row above the cutover (1280 yaml cap + judged thinking).
- qwen rows above the cutover (sampling recipe incomplete).
- LFM `humaneval_instruct` rows (ALL eras, incl. post-cutover: the primed-fence format needs its stop-strings; eos-only breaks extraction). Authoritative LFM HumanEval = task `humaneval_think` rows.
- Rows tagged `smoke_*` (probes, never results).
- `gptoss_20b_high`/`gptoss_120b_high` humaneval rows at max_gen_toks 2048 (purged).

## Variant-model records (adaptation program)
`gemma4_adapted` and `gemma4_ctrl_sft` (fine-tune trio) and `lfm25_vllm` (routing-fix
era pair) are greedy/E2-era rows: internally paired within their runs, valid ONLY as
within-trio comparisons (01-findings carries the era note), never mixable with E4/E5
levels. `lfm25_vllm` is fully superseded by later `lfm25_instruct` rows.

## Column caveat
The CSV's `max_gen_toks` column records the BASE budget; the hard ceiling is the
driver's `--backoff-cap` (2048 default; 4096 thinking; 8192 ifeval-thinking), which is
not recorded per-row. Cap identification: per-cell `[backoff]` lines in the run logs
and token-count distributions in the dumps.

## Known measurement limitation (documented, not fixed)
Think-length arrays (`analysis_toks`/`raw_think_toks`) include backoff-retry
re-generations (unaligned to doc ids), oversampling long thinkers at arm-dependent
rates. think_analysis.py restricts exact claims to cells whose array length matches the
item count; other cells are approximate.

Producer of the corrected era: `analysis/residency/instruct_genbench_vllm.py` at commit
`3f19416`+; chains committed under `scripts/residency/` (`final_reruns.sh`,
`final_reruns_tail.sh`, `takeover.sh`, `high_gsm8k_fix.sh`, `audit_fixes.sh`). Defect
history: docs/research/mechanism/02-corrections.md §6.
