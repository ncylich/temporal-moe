# Protocol eras — instruct benchmark results (CURRENT STATE 2026-08-14)

## Selection rule (the only one)
`instruct_genbench_vllm.csv` contains AUTHORITATIVE ROWS ONLY: exactly one row per
(record, arm, task, metric), all produced by the final single-pass protocol or by a
bespoke producer listed below. There is no in-file history and no cutover marker.
Full history (every superseded, probe, and invalid row, original order):
`superseded/instruct_genbench_vllm_history.csv`.

## The final protocol (E7, "single-pass")
- One generation pass per request at the cap (`--gen-cap`; recorded truthfully in the
  `max_gen_toks` column): 2048 non-thinking, 4096 thinking, 8192 thinking-IFEval.
  Responses finishing at the cap are counted and scored as-is (degeneracy suspects).
- Sampling: each model's card recipe (incl. presence_penalty, per-mode temp/top_p),
  seed 1234; no-recipe fallback 0.7/0.95. Never greedy.
- Stops: eos-only at the engine; think segment stripped (per-arch marker), then task
  stops applied (`genprotocol.py`).
- Capture: doc-keyed raw token dumps + think lengths (`genbench_tokens/`,
  `genbench_samples/`).

## Bespoke producers (valid, non-lm_eval budgets)
- `humaneval_gemma_fixed` (1536 think-off / 3072 think-on): `humaneval_gemma.py`.
- `humaneval_gptoss` (2048 low/med, 4096 high): `humaneval_gptoss.py`.
- `humaneval_think` (4096): `humaneval_think.py` (LFM, qwen think-on).
- `mmlu_gptoss_relaxed`: `mmlu_gptoss.py` (harmony-tolerant extraction).
- `,answer-only` rescores: `rescore_answer_only.py` (max_gen_toks column reads
  "rescored").
- `gemma4_adapted`/`gemma4_ctrl_sft`: greedy-era adaptation trio — valid ONLY as the
  internally-paired three-way comparison (era note in 01-findings §5).
- `lfm25_fullset_audit`: full-541 IFEval parity probe (ladder era, 1024 base) — audit
  context only.

## Known-invalid task/record combinations (never live, by construction)
`smoke_*`, `lfm25_vllm`; `humaneval_instruct` for lfm/qwen-think-on/gemma/gpt-oss
(primed fence breaks thinking/channel templates); `mmlu_flan_cot_fewshot` for gpt-oss
(stock extraction floors harmony answers). Enforced by `partition_eras.py`.

## Era history (abbreviated; details in 02-corrections §6 and reroll_delta_record.md)
E1 greedy → E2 sampled (judged thinking text; yaml budget caps) → E3 partial fixes →
E4 native-path corrections → E5 sampling-fallback fix → E6 no-re-roll (continuation)
→ E7 single-pass uniform (final). Ladder eras (≤E6) systematically understated scores
(mean +2.96 on re-measurement): `reroll_delta_record.md`. Self-CE and all
teacher-forced results were never affected by any generative-era defect.
