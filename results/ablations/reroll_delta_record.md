# Ladder-era vs clean-era deltas (2026-08-14)

Paired comparison of every cell measured both under the truncate-and-retry ladder
(regeneration or continuation) and under the final single-pass protocol (identical
sampling, budgets, scoring). n=53 pairs, all post-cutover.

**Verdict: the ladder eras systematically UNDERSTATED scores. Reproducible form
(from the two CSVs, any-budget post-cutover pairing): n=160 pairs, mean +2.5, mean
|delta| 2.9, max +14.5 (LFM R4 IFEval); direction uniform-positive. (An interim
in-session pairing reported n=53/+2.96; that exact pairing is not reproducible from
the files and is superseded by the numbers above.) Direction was uniform-positive across models and arms, so paired
damage numbers moved less than levels, but several materially (LFM IFEval damage
-8.0 -> -2.0; gemma think-on IFEval free 0.825 -> 0.925).

Mechanism (working hypothesis, not fully decomposed): re-rolled retries discarded
partially-complete trajectories and redrew fresh ones whose think-length distribution
re-exposed them to truncation; single-pass commits one trajectory with the full budget.
The hypothesized score-inflating retry bias was not observed; the truncation-replay
harm dominated.

All ladder-era rows live only in superseded/instruct_genbench_vllm_history.csv (verified after the 2026-08-14 final partition).
Producer of pairs: session log 2026-08-14 ~14:55 UTC; live rows:
instruct_genbench_vllm.csv (authoritative-only).
