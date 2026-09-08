# Instruct thinking-ablation plan (2026-08-12)

Hypothesis: thinking protects against residency damage (length-damage Spearman +0.72 is
confounded); mechanism candidate: constrained decode degrades reasoning progress per
token, so a confidence-gated thinker needs more tokens to conclude.

## Runs (n=200/task; GSM8K, IFEval, HumanEval, MMLU; sampled protocol*)
| model | default mode (sampled grid) | opposite arm (ablation chain) | arms |
|---|---|---|---|
| gemma4-26B-IT | think OFF | think ON (`enable_thinking=True`) | free, R8, R16 |
| Qwen3.5-35B | think ON | think OFF (template kwarg, else empty-think prefill) | free, R8, R32 |
| gpt-oss-20b | effort medium | effort LOW and HIGH | free, R4 |
| gpt-oss-120b | effort medium | effort LOW and HIGH | free, R4, R16 |
| LFM2.5-A1B | think ON (fixed: no toggle exists; LFM2-24B is a different model) | — excluded from pair-stack | free, R4 |

*Sampling from each model's generation_config (never greedy), seed 1234, truncation
backoff x2 to hard cap 2048 (cap-outs logged as degeneracy suspects, scored as-is).
gemma HumanEval runs channel-aware (humaneval_gemma.py) in both modes: thinking-channel
creation disabled when OFF, enabled when ON. gpt-oss scored on final channel only.

## Captured per item (all cells)
- Generated token ids (re-tokenized from text; approximation accepted) ->
  /workspace/instruct-traj/genbench_tokens/ (workspace + HF mirror; never in git)
- gen_toks, think_toks (think-span tokens), backtracks (wait/actually/hmm/re-check
  count in think spans), benchmark score -> results/ablations/genbench_samples/
- Cells finished before capture wiring (OLMoE): cheap tail rerun re-dumps tokens.

## Analyses (post-run; no new experiments without sign-off)
1. Damage x thinking mode: (constrained - free) within each mode, per model x task.
   Thinking-protection predicts smaller damage with thinking ON; gpt-oss low/med/high
   gives the dose-response version.
2. Think-length shift: think_toks constrained vs free per model x task. Throughput
   mechanism predicts constrained > free, growing with tightness (R8 vs R16; R4 vs R16).
   LFM included here despite no off-arm.
3. Backtracks per think-token: flat rate + more tokens = uniform dilution (thinking
   slower); elevated rate = error reaction (thinking worse). Not mutually exclusive.

## Timeline (single H100, queued)
- Sampled grid (default modes, all 6 models): done ~10:00 UTC.
- Ablation arms (~16 arm-suites incl. two 120b restages): done ~16:30 UTC (~9:30am PT).
- Analyses + figures: ~1h after data lands.

## Storage policy
No large binaries in git (repo carries code, CSVs, figures only; largest tracked file
13MB). Token dumps and any future big artifacts: workspace disk + HF hub.
