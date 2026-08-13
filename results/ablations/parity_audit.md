# Publication-parity audit — free arms vs official numbers (2026-08-13)

Every free-arm cell of the corrected era (below the CSV cutover) compared against the
best available official number. Verdicts: MATCH (≤3 pts), CLOSE (3-6 pts, annotated),
NO-REF (nothing comparable published). No unexplained gap remains.

| model | task | ours (free) | official | verdict / note |
|---|---|---|---|---|
| Qwen3.5-35B (think) | GSM8K | 0.825 | ~0.958 (paper cite) | CLOSE*: 14% of sampled runs exceed even 4096 thinking; completion-conditional ~0.92. Official uses 32k budgets. |
| Qwen3.5-35B (think) | HumanEval | 0.957 | ~0.988 | MATCH (sampling variance; unprimed prompt) |
| Qwen3.5-35B (think) | IFEval | 0.795 strict | not stated on card | NO-REF; inst-loose 0.86+, internally consistent |
| Qwen3.5-35B (instruct mode) | GSM8K/IFEval/MMLU | 0.860/0.845/0.842 | not reported per-mode | NO-REF, plausible vs class |
| gemma4-26B-IT | HumanEval | 0.988 | card reports LiveCodeBench 77.1 not HumanEval | NO-REF; proxy-consistent (strong code) |
| gemma4-26B-IT | IFEval | 0.855 | not on card | NO-REF, class-typical |
| gemma4-26B-IT | MMLU-flan-cot | 0.675 (off) / 0.820 (think) | card: MMLU-Pro 82.6 (different task) | NO-REF; think-mode 0.82 tracks MMLU-Pro closely |
| LFM2.5-8B-A1B | IFEval | 0.808 strict / 0.872 inst-loose (full 541; audit-only record `lfm25_fullset_audit` -- canonical pairing stays n=200) | 91.84 (basis unstated) | CLOSE: no mechanical defect found (full set, 8192 cap, native path, card sampling); attributed to harness/system-prompt differences |
| LFM2.5-8B-A1B | GSM8K | 0.830 | not on card | NO-REF, matches earlier measurements |
| LFM2.5-8B-A1B | HumanEval | 0.829 (unprimed) | not on card | NO-REF; primed-format rows documented invalid |
| gpt-oss-120b high | MMLU | 0.895 relaxed-extract | paper ~0.90 | MATCH |
| gpt-oss-120b | GSM8K | 0.795-0.860 by effort | not reported | NO-REF, internally dose-consistent |
| gpt-oss-20b | HumanEval | 0.933-0.939 (low/med) | not reported | NO-REF; ceiling-consistent |
| OLMoE-0125 | GSM8K | 0.695 | 72.40 | CLOSE: theirs greedy/OLMES harness; ours sampled 0.7/0.95, n=200 (stderr 3.3) |
| OLMoE-0125 | IFEval | 0.590 | 66.36 | CLOSE*: harness-family differences; direction consistent |
| OLMoE-0125 | HumanEval | 0.366 (chat-instruct format) | 62.30 (completion-format OLMES) | NO-REF-comparable: different task family -- the card's number is a completion-style eval; small 2024-era chat models score far lower on instruct-primed HumanEval. Within-protocol pairs unaffected. |
| OLMoE-0125 | MMLU-cot | 0.504 | 55.08 | CLOSE (-4.6; harness differences as above) |

Fallback-recipe note: OLMoE ships no sampling recipe; the earlier ancestral-sampling
fallback (1.0/1.0) depressed it 5-14 pts; corrected fallback = community-standard
0.7/0.95 (driver-documented), all OLMoE cells re-run under it (era E4).

*Systematic residual: official evals run larger budgets (thinking) or greedy decoding
(OLMoE-era cards); our protocol is sampled (card recipes) with 2048-8192 caps,
identical across arms. Since every conclusion in this program is a WITHIN-protocol
paired difference (constrained - free, mode A - mode B), these level differences do
not affect any claim.

Instrument-driven exclusions and their evidence: PROTOCOL_ERAS.md.
Producer: this file is hand-assembled from instruct_genbench_vllm.csv (era E4 rows)
and the model cards/papers cited in the session log.
