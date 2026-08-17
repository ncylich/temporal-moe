# Gemma4-26B-A4B-IT rolling-residency adaptation — RESULTS

Final result of the 2026-08-16/17 adaptation program: make gemma4-26b-it robust to the
rolling-residency serving constraint (R resident experts/layer, ≤1 swap/token, min_logit
eviction, prefill free, rule on generated tokens; tight arm R8 = k).

## Headline

**D12** — constraint-aware CE on the model's own compliant trajectories (d7 pool) with a
KL-to-base anchor at weight 0.05, 3.4M response tokens — essentially eliminates the R8
constraint penalty. Damage vs the *unconstrained* base (percentage points, authoritative
200-item instrument; negative = worse than base-free):

| R8 arm            | GSM8K | IFEval | HumanEval | MMLU  |
|-------------------|-------|--------|-----------|-------|
| base (unadapted)  | −6.0  |  0.0   | −6.1      | −0.2  |
| **D12 (adapted)** | **0.0** | **−1.0** | **−1.2** | **−1.8** |

Same-arm deltas (D12 minus base at the same constraint): R8 GSM8K +6.0, IFEval −1.0,
HumanEval +4.9, MMLU −1.1; free arm +0.5 / +1.5 / −1.8 / −2.8. MMLU deltas use multi-run
means (runs listed below); all other cells are single authoritative runs.
Residual weakness: a 2–3pt MMLU cost in the free arm (base free 94.3 → D12 free 91.5,
3-run mean over 93.4/91.2/89.9 — free-arm MMLU has ~3.5pt run-to-run spread at temp 1.0).

## Recipe (all settings load-bearing)

- Data: d7 prompt pool — 9,173 benchmark-free prompts (domain8k 4,958 incl. 431 code rows;
  mcq-writer 691; mathlane_v2 2,341; d5 few-shot variants 1,183). Lineage rule: no
  benchmark-family data in any form (test/train splits, synthetic derivatives); pools
  8-gram-screened vs GSM8K/MMLU/HumanEval/IFEval test sets.
- Trajectories: the model's own think-off responses (sampling per its generation config,
  seed 1234), generated under NO constraint, 2048-token cap, truncation gate <15% capped.
- Training: expert-tensor LoRA r16 on the fused 3D expert tensors (grouped_mm path) +
  attention LoRA r32; constraint ON during CE (R8, per-row enforce_from = prompt length,
  batched rows); micro-batch 2 / seq 4096 / 16 rows per step; lr 3e-5.
- KL anchor: weight 0.05 against precomputed base top-50 free-routing logprobs on response
  tokens; CE backward runs before the KL forward (one live graph); KL scoring checkpointed.
- Budget: 3.4M response tokens. Producer: analysis/residency/train_gemma_ce.py.

## Why these settings (the ladder)

1. **Constraint-aware beats plain self-SFT** (E6 control): same data, constraint off during
   training → the gains disappear.
2. **KL anchor strength is a free-arm/constrained-MMLU dial**: no KL (D7) → R8 MMLU −0.7 but
   weak free arm; KL 0.1 (D8) → free arm repaired (only positive free-MMLU cell of the
   program) but R8 MMLU −2.9; **KL 0.05 (D12) interpolates**: free arm held, R8 MMLU −1.1,
   and the strongest constrained row of the program.
3. **More tokens hurt**: at 10M tokens the KL-0.1 recipe collapses (constrained GSM8K
   +4 → −10, D10) while the no-KL mix merely redistributes (D11). The KL×duration
   interaction is toxic; 3.4M is the operating point. (D10 carried a mid-run optimizer
   restart after a quota crash; loss recovered fully, but a crash-free 10M run would be
   needed to fully deconfound.)
4. **Benchmark lineage is a trap**: Orca-Math (GSM8K-train-seeded) produced a fake +8 GSM8K
   that vanished when the lane was removed (D1 vs D4 ablation) — style-matching, not
   constraint robustness. Hence the strict lineage ban above.
5. **Think-on needs a bigger envelope**: on this pool, 35.7% of think-on responses exceed a
   3,072-token cap (median think response 2,346). A valid think-on run needs ≥6k generation
   budget and a training-memory rework (8k seqs don't fit mb2); not run.

## Measurement discipline

- Screeners (active-item subsets, screening_genbench.csv) are RELATIVE instruments:
  constrained arms are batch-composition sensitive (up to 8.6pts on IFEval R8 between
  70-item and 200-item batches). Candidates compare only against same-batch base references;
  winners get a full authoritative grid (instruct_genbench_vllm.csv) before citation.
  Screening deltas overstate full-instrument deltas (hard-item amplification): D8's
  screening +4 GSM8K R8 compressed to +1 authoritative; D12's +6 held at +6.
- MMLU is dual-scored from the same generations (strict "The answer is (X)" vs relaxed
  extraction); strict measures few-shot format imitation, not knowledge — relaxed is the
  reported metric. mmlu_gptoss.py, extractor v2.

## Records

Authoritative: gemma4_ce_d12 + gemma4_ce_d12_dual/_dual2 (and gemma4_ce_d8 + duals) in
instruct_genbench_vllm.csv; base = gemma4_instruct rows + dual_base/pair_base means.
Screening ladder: scr_d5..scr_d12 (+_dual) in screening_genbench.csv. Adapters:
/workspace/olmoe-adapt/data/gemma_ce_d{5,7,8,9,10,11,12}_adapter.pt (d12 = the result).
Think-on trajectory file for a future resized run: /workspace/instruct-traj/gemma4_d7think.pt.

## Qwen3.5-35B-A3B replication (2026-08-17)

The recipe transfers. Same pipeline on qwen (think-off, same-batch refs; records
qwen35_ce_d12r / qwen35_val_base + duals in screening_genbench.csv): base R8 damage
−9.5/−12.5/−2.5/0.0 (GSM8K/IFEval/HE/MMLU) becomes −3.0/−9.0/+0.6/−2.2 adapted —
same-arm gains +6.5/+3.5/+3.0/−2.2, the gemma signature (constrained recovery, small
MMLU cost) at matching magnitudes. Divergences: free-arm IFEval −5.5 (gemma gained);
damage not fully eliminated. Single run. Documented accommodations forced by 70GB
weights on an 80GB card: expert-LoRA r8 (capacity-matched to gemma's 1.4B), paged
8-bit Adam, HF stack (unsloth's batched constrained path drifts 4.9% on qwen where
plain HF shows 0.0–0.3%), chunked-checkpointed CE, per-row KL forward, cuDNN SDP off.

## Open

- Free-arm MMLU cost (−2.8): untested lever = KL bracket 0.03/0.07.
- Qwen free-arm IFEval regression (−5.5): unexplained; candidate suspects are the
  r8/8-bit accommodations or qwen's larger baseline constraint damage.
- Think-on variant: needs ≥6k generation cap + training-memory rework (35.7% of
  think responses cap at 3072 on this pool).
