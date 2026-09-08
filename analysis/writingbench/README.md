# WritingBench harness (residency arms)

Minimal local setup for scoring our models on WritingBench (arXiv 2503.05244)
under the rolling-residency constraint. Fully offline: generation through the
project's constrained vLLM stack, scoring through the official critic model.

- `upstream/` — clone of X-PLUG/WritingBench (1,000 queries, 555 English, 5
  instance-specific criteria per query in `checklist`; scoring prompt in
  `prompt.py`).
- `critic-model/` — AQuarterMile/WritingBench-Critic-Model-Qwen-7B (local judge).
- `venv/` — orchestration venv (system-site over /opt/venv_vllm, so vLLM is
  inherited; only extras installed here).
- `wb_generate.py` — responses for the first N English queries under an arm:
  `--model-path ... --record qwen35_base --arm R8 --n 50 --think off`.
  Boots vllm_glue + vllm_residency + DEC exactly like the genbench drivers.
  Sampling 0.7/0.8 seed 1234, 2048-token cap (long-form caveat: WritingBench
  official numbers use longer budgets; ours are for free-vs-constrained deltas,
  not leaderboard comparison).
- `wb_score.py` — critic scoring, 5 criteria per item batched in one vLLM pass;
  per-item mean of criteria, record mean ± SE to `scores/summary.csv`.
- `smoke.sh` — 10-query end-to-end check.

Protocol notes: same-boot/same-batch discipline applies as everywhere in this
project; compare records generated with identical `--n` and settings only.
Scores are 1-10 critic points, higher better; the interesting quantity is the
delta between arms of the same model, not the absolute.
