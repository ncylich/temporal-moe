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

## Cells above the cutover that remain valid (no E4 rerun needed)
- OLMoE all cells: non-thinking, 640-token answers -- no defect ever bound (E4 re-runs
  them anyway for token capture; either era's scores agree).
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

Producer of the corrected era: `analysis/residency/instruct_genbench_vllm.py` at commit
`3f19416` or later; chain `final_reruns.sh`. Defect history: docs/research/mechanism/
02-corrections.md (entry pending) and the session log.
