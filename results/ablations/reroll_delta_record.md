# Ladder-era vs single-pass deltas (2026-08-14)

What changed when the truncate-and-retry ladder (regeneration, then continuation)
was retired for the single-pass-at-cap protocol: every cell's last post-cutover
ladder-era measurement paired with its single-pass replacement, primary accuracy
metric per task.

Producer: `analysis/residency/reroll_delta.py` (committed). Both CSVs are read at
pinned commit `dba0c2e` (after the full single-pass re-measurement, before the
2026-08-14 gpt-oss 8192-budget wave), so these numbers reproduce regardless of
later data movement. Output, verbatim:

    ALL: n=59  mean +2.60  mean|d| 3.12  max +14.5 (lfm25_instruct R4 ifeval)  positive 39/59

**Verdict: retiring the ladder RAISED scores on net — mean +2.6 points, majority
positive (39/59 pairs), max +14.5 (LFM R4 IFEval) — i.e. the ladder eras
systematically understated performance.** Individual pairs move both ways
(20/59 negative), consistent with sampling noise around a positive shift; the
claim is about the mean, not each cell. Because direction was broadly shared
across arms, paired damage numbers moved less than levels, but several materially
(LFM IFEval damage -8.0 -> -2.0; gemma think-on IFEval free 0.825 -> 0.925).

Caveats, stated plainly:
- A same-budget pairing is impossible: ladder rows record the base budget,
  single-pass rows record the cap, so every pair differs in the recorded
  `max_gen_toks` (the script's same-budget stratum is empty). The delta therefore
  bundles the ladder-mechanism change with the budget-accounting change.
- Post-cutover history rows cannot be era-attributed from the CSV alone; a few
  pairs may reflect bad-run replacement rather than ladder retirement.
- An earlier in-session version of this record claimed n=53 pairs / mean +2.96
  under "identical sampling, budgets, scoring"; that pairing was never
  reproducible from the files and is retracted in favor of the script above.

Mechanism (working hypothesis, not fully decomposed): re-rolled retries discarded
partially-complete trajectories and redrew fresh ones whose think-length
distribution re-exposed them to truncation; single-pass commits one trajectory
with the full budget. The hypothesized score-inflating retry bias was not
observed; the truncation-replay harm dominated.

All ladder-era rows live only in `superseded/instruct_genbench_vllm_history.csv`
(verified after the 2026-08-14 final partition).
