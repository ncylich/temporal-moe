# Proposal: rewrite of 01-findings.md section 5 (and touched lines in 1, 6, 7, 8)

Proposal only. No doc edited. Every number below was read from the named CSV in this
sitting, not from a doc. Branch context: correct-convention sources are on
`layer-lexicality` through commit `326effa` plus pod-local rows listed in part 5.

**Sections 1–4 verification, not inheritance.** The FLAME phase-0 impose cells route
through Megatron's `topk_softmax_with_capacity` (softmax over the selected top-k), so
renormalising over the selection is those models' NATIVE convention and pre-softmax
masking reproduces it exactly. The gate-mass defect was specific to masking a
`norm_topk_prob=False` model (OLMoE) with renormalisation it never had. Sections 1–4's
numbers stand. One sentence does not: the section 1 depth bullet "This is learned, not
mechanical. Imposing the constraint without training produces no shift at all" cites
section 5's renorm-era locus null, which inverted. Its replacement is in part 2.

## 1. Claim-by-claim triage, sections 5–8

| # | Claim / number in 01 | Verdict | Correct value | Source (read this sitting) |
|---|---|---|---|---|
| 1 | Locus table: impose −0.0041, CE +0.0932, CE+PLE +0.0964 | **corrected, sign inverted** | base free −0.1445; impose (no training) **+0.1001**; distill-adapted +0.1032 | `ple_locus.csv` rows `locus_base_free`, `locus_impose_R8_preserve`, `locus_distill100M_R8` |
| 2 | "The contextual shift is learned, not imposed" | **inverted** | imposing produces the whole shift (+0.24 of +0.25); adaptation adds +0.003 | same rows |
| 3 | "Per-layer embeddings add nothing" (0.0031 vs spread 0.0031) | void, era rows (`p4_*` block carries the −0.0041 reference) | PLE is dead under preserve on stronger grounds: 0.8104 zero-init, 0.8061 calibrated, vs LoRA 0.7887 | `sweep_RESULTS.md` finding 12; calibrated table `ple_calib_meta_preserve.json` |
| 4 | Five adaptation nulls (PLE-on-LoRA 0.49σ; calibrated init 1.23σ worse; sequential vs joint 0.47σ; LoRA rank r=32 1.31σ worse) | **void, era records**; calibrated init re-ran and **flipped sign** (now helps: 0.8061 < 0.8104 zero-init, still dead vs 0.7887) | pattern-4 paragraph; only the calibrated arm has a preserve re-measurement | finding 12; `/tmp` night logs pending commit (part 5) |
| 5 | "Single-layer damage is U-shaped, worst at layer 1, lowest at layer 11" | **inverted** | monotone in depth: L15 +0.0223 is 3.3× the interior mean; L14 +0.0114 second; L01 +0.0059; L11 +0.0078 is unremarkable | `olmoe_gatemass_remeasure.csv` solo rows |
| 6 | Figure `results/phase0/figures/layer_freeing_damage.png` | plots the artifact; misfiled (OLMoE under phase0) | replace with `results/ablations/figures/olmoe_perlayer.png` (committed, producer `plot_scaling.py` figure3) | file exists on branch |
| 7 | "Layers 2 and 15 tie on solo damage, 0.1408 vs 0.1408" | void; artifact of the ~2.5× scaling | under preserve they differ 4.7×: L02 +0.0048, L15 +0.0223 | `olmoe_gatemass_remeasure.csv` |
| 8 | "Training-free profile predicted layer 2 at 5.8× layer 15" | void; the prediction itself was the artifact | corrected profile ranks L15 first, L14 second — the order the trained outcomes reward | same |
| 9 | Free-set table {0,1}/{0,1,2}/{0,1,15}/{0,1,14,15}: BPB 0.8144/0.8086/0.7978/0.7863, ds 0.5937/0.5937/0.6030/0.6037 | **era-tainted**: cells trained and scored under renorm; `layer_freeing_downstream.csv` later re-based the references but not the cells | preserve replacements in part 2: training-free joint table (`olmoe_freeset_joint.csv`) + trained {14,15} cell 0.7600 BPB / 0.6119 ds | `olmoe_freeset_joint.csv`; finding 12 (row pending commit, part 5) |
| 10 | **"Do not choose free sets from single-layer damage"** | **inverted — the most important call, argued in part 2** | under preserve, solo damage picks the winning sets at matched memory | `olmoe_freeset_joint.csv`: top4-solo {10,12,14,15} +0.0936 vs head {0,1,2,3} +0.1418 vs inherited {0,1,14,15} +0.1092 |
| 11 | Adaptation-strategy table (router-only 0.707; +norms 0.914; +LoRA 0.914; full FT 0.934; anneal/self-distill nulls) | **void, era records, no preserve re-run**; the recovery *scale* is era-dependent: recomputed on preserve anchors, LoRA-15M recovery is 1−(0.7887−0.6727)/(0.8393−0.6727) = **0.30**, not 0.91 | no preserve bake-off exists; re-run estimate in part 5 if wanted | anchors: `olmoe_remeasure` free 0.6703/0.6727 (slice variants), impose 0.8393, adapted `frontier_olmoe.csv` smoke 0.7779 / campaign 0.7887 |
| 12 | "Cheap adaptation gets within 0.02 of a full fine-tune"; LoRA rank sweep (r=8 0.893, r=64 0.910) | void with #11; rank sweep never re-run | drop; correct-era record covers r=32 only | — |
| 13 | Per-task downstream table (impose: lambada 0.000, arc_easy 0.280, "catastrophic") | **inverted in severity**: catastrophic was the ~2.5× block-output scaling, not the constraint | preserve impose is mild: arc_easy 0.6364 vs free 0.7698; hellaswag 0.4864 vs 0.5847 | `olmoe_downstream_ref.csv` (producer `make_downstream_ref.py`) |
| 14 | "Adaptation recovers about 70% of the gap" (0.675/0.698/0.699) | void with #9/#13; preserve analogue is finding 7's 32% downstream recovery | 0.6119 mean ds for trained {14,15} vs impose floor 0.5978 and free 0.6727 (values re-read from `olmoe_downstream_ref.csv` aggregation) | `layer_freeing_downstream.csv` re-based rows |
| 15 | §6 bullet "Do not pick free sets from single-layer ablation. It is wrong on this model in three cells." | inverted with #10 | new wording in part 2 | — |
| 16 | §6 bullets on demand oracle and smoothing | **survive** — phase-0 measurements, convention-clean per the §1–4 verification | — | `e5_eviction_policy_headroom.csv`, `e7_demand_smoothing.csv` untouched |
| 17 | §7 "No per-layer measurement above 13 layers on models we trained. The 16-layer evidence … a different program" | superseded by new data | per-layer d_l(R) now exists at 48, 40, 30 layers on three external models | `perlayer_qwen3.csv`, `perlayer_qwen3_5.csv`, `perlayer_gemma4.csv` |
| 18 | §8 bullet 1 (forecastability vs freeing, "open by construction") | partially answered; rewrite | solo damage ranks freeing correctly on OLMoE under preserve; forecastability-vs-freeing cross-model still open | part 2 wording |
| 19 | §8 bullets 2–3 (1e19 re-run, sham producer) | survive untouched | — | — |
| 20 | §1 depth bullet "learned, not mechanical … Section 5" | **inverted** with #2 | new wording in part 2 | `ple_locus.csv` |

Stale-value carriers beyond 01 (grep of `0.1408`, `5.8`, `U-shaped`, `layer 11`,
`0.814440/0.797810/0.786275`, `worst at layer 1`): `05-notebook.md` (history doc —
keep, it narrates the era; 02 carries the correction), `archive/01-findings-superseded.md`,
`archive/LAYER_LEXICALITY.md`, `archive/LAYER_LEXICALITY_ROUND2.md` (era records —
exempt by decision 4), `results/ablations/layer_freeing_RESULTS.md` (**live file, era
numbers, no void banner — needs the archive treatment or a banner; open question 3**),
`results/ablations/README.md` (one line citing the U-shape figure — fix in the same
commit as 01), `results/ablations/qwen35_RESULTS.md` (hit is an unrelated "layer 11"
row label — no action).

## 2. Proposed section 5, and the touched lines elsewhere

Outline (claim-first, one table per claim, every number from a committed CSV):

**5. Adapting a pretrained model** *(all numbers gate_mass=preserve; the renorm-era
measurements this section replaces are archived in `results/archive/olmoe_wrong_renorm/`
and narrated in 02-corrections.md)*

1. **"Imposing the constraint contextualises routing by itself; adaptation adds almost
   nothing."** Table from `ple_locus.csv`:

   | condition | context minus token |
   |---|---|
   | base model, free routing | −0.1445 |
   | constraint imposed, no training | +0.1001 |
   | adapted (distill, 100M tokens) | +0.1032 |

   With the sentence: the base OLMoE router is the most lexical in the program (token
   AUC 0.837), and masking to the resident set mechanically forces selection off the
   token axis. Cross-model confirmation one line, pointing at `locus_qwen.csv`
   (qwen3.5: −0.0002 free → +0.1265 imposed).

2. **"Single-layer damage rises with depth; the last layer is 3.3× the interior mean."**
   Replacement figure `olmoe_perlayer.png`. Table: solo damages summarised (L15 +0.0223,
   L14 +0.0114, interior mean +0.0067, L01 +0.0059), source `olmoe_gatemass_remeasure.csv`.

3. **"Solo damage picks the right layers to free."** The inversion of the old punchline,
   stated with its evidence and its limit:

   | free set (4 of 16, matched memory) | joint damage, BPB |
   |---|---|
   | top-4 by solo damage {10,12,14,15} | **+0.0936** |
   | tail {12,13,14,15} | +0.1019 |
   | renorm-era pick {0,1,14,15} | +0.1092 |
   | head {0,1,2,3} | +0.1418 |

   Source `olmoe_freeset_joint.csv` (training-free, blocked spread ±0.016). Trained
   confirmation: the {14,15} free-set + distill run is the best adapted OLMoE cell in
   the program, BPB 0.7600 against 0.7887 all-constrained, downstream 0.6119 against
   0.6017 (row pending commit — part 5). Limit stated plainly: no *trained* controlled
   pair ({0,1,15} vs {0,1,2} style) exists under preserve; the training-free joint
   table carries that comparison alone (open question 1 proposes the 2-hour re-run).

4. **"Imposition is mild, not catastrophic."** Per-task table from
   `olmoe_downstream_ref.csv` (free / imposed / adapted columns, five representative
   tasks), with the sentence naming the old catastrophe as the scaling artifact.

5. **Adaptation under the correct convention** — replaces the strategy bake-off with
   what is actually measured: distillation at r=32 LoRA recovers 0.30 of the BPB gap
   and 32% of downstream (finding 7 numbers, `frontier_olmoe.csv` +
   `olmoe_downstream_ref.csv`); per-layer allocation beats uniform untrained
   (fitted_B192 0.7902 vs uniform 0.7952, `frontier_olmoe.csv` alloc rows) and
   converges once adapted (0.7585 vs 0.7601). One pattern-4 paragraph for the
   strategy bake-off, the anneal/self-distill/sequential nulls, and the LoRA rank
   sweep: era records, none stand, ideas untested under the correct convention,
   archive pointer.

6. **PLE, one paragraph** (pattern already used in FINDINGS.md): dead under both
   inits under preserve (0.8104 zero, 0.8061 calibrated, vs LoRA 0.7887); the
   calibrated-init *sign* corrected relative to the era record.

**Touched lines elsewhere:**
- §1 depth bullet becomes: "**Imposed on a pretrained lexical router, the shift is
  mechanical** (+0.24 without any training, section 5). Whether the trained-from-scratch
  gap is additionally learned is open; no phase-0 impose-locus measurement exists."
- §6 free-set bullet becomes: "Pick free sets from single-layer damage measured under
  the model's own gate convention. The published contrary advice was measured under a
  broken one."
- §7 one-model caveat becomes a pointer: per-layer damage profiles now exist for
  Qwen3-30B (48 layers), Qwen3.5-35B (40), gemma4-26B (30) — `perlayer_*.csv` — and the
  cross-model law lives in `results/ablations/sweep_RESULTS.md` (see part 3).
- §8 bullet 1 becomes: "Does demand forecastability predict freeing value *cross-model*?
  Solo damage now ranks freeing correctly on OLMoE; the forecastability correlate has
  never been measured on any adapted model."

## 3. What moves from miscellaneous_findings.md into 01, what stays

**Moves into 01 §5** (adapted, not verbatim — the target voice drops the "program log"
framing): the "Per-layer residency relaxation" section — d_l(R) profile, greedy
allocation numbers (0.7902/0.7952 @192; 0.7644/0.7687 @256), adapted-surface
convergence, free-set {14,15} result, figure link. These claims then have exactly one
home, in 01. miscellaneous_findings.md keeps a one-line pointer.

**Stays in miscellaneous_findings.md:** §3 (32% downstream recovery framing vs
OLMo-1B) and §4 (token-efficiency crossing), which are adaptation-program results
rather than mechanism results, and the PLE obituary paragraph. 01 §5 points at the
obituary instead of restating it, so the negative keeps one home.

**Cross-model data placement (the §7 question): recommend no new section in 01.** The
granularity law, ladder, frontier, and instruct results form a program with its own
committed record, `sweep_RESULTS.md` findings 1–12, and their claims already have that
home. 01 §5 is OLMoE mechanism evidence and cites `perlayer_*.csv` in one sentence, as
replication of the depth profile. Pulling the cross-model program into 01 would
duplicate twelve findings or orphan them. The routing index (new README, decision 1)
gets a row for `sweep_RESULTS.md` so the program is findable.

## 4. 02-corrections.md entries required

1. **Locus inversion.** Era measurement said imposing moves nothing (−0.0041); the
   masked distribution was scaled ~2.5×, which saturated the probe. Under preserve,
   imposition produces the whole shift (+0.24) and adaptation adds +0.003. The
   "contextual shift is learned" claim, and §1's dependent bullet, reversed.
2. **Per-layer profile inversion.** U-shape worst-at-L1 → monotone-in-depth worst-at-L15
   (3.3× interior). The published figure plots the artifact; retired to the archive.
3. **Free-set punchline inversion.** "Do not choose free sets from solo damage" rested
   on solo ranks that were themselves artifacts. Corrected ranks pick the winning sets;
   the era's controlled pair is void; the trained {14,15} preserve cell is the program's
   best.
4. **Imposition severity.** Lambada-to-zero was the scaling artifact, not the
   constraint. Preserve imposition costs +0.169 BPB and single-digit downstream points.
5. **Adaptation recovery scale.** 0.91 recovery was measured against a catastrophic
   denominator; preserve recovery is 0.30 BPB / 32% downstream. The strategy bake-off
   and its nulls are era records; only the calibrated-init arm was re-run (sign flipped:
   calibration now helps slightly; PLE still dead).
6. **Attribution note**: the era's within-era *comparisons* (e.g. LoRA ties norms) may
   still hold direction but have no preserve measurement; none is quoted in 01.

## 5. Commit-first list and runs needed

Pod-local, must be committed before the rewrite cites them (all cheap):
1. **Trained free-set {14,15} cell has no CSV row anywhere.** The program's headline
   preserve result exists only in `sweep_RESULTS.md` prose and `/tmp` night logs.
   Re-score BPB from the `at10M` surface (~3 min GPU) and transcribe the downstream
   row with producer, or commit a small `olmoe_freeset_trained.csv` from the logs
   with the snapshot caveat. **Defect, blocks part 2 item 3.**
2. **PLE preserve-era 15M arms** (0.8104 / 0.8061) likewise live in logs and one
   sweep_RESULTS line; add rows beside them in the same CSV.
3. `instruct_genbench.csv` new rows, `instruct_genbench_vllm.csv`, `genbench_samples/`
   — commit when the running grid completes (not cited by 01, but part of the same
   hygiene pass).
4. `wildchat_prompts_500_meta.json` (sha pin) is uncommitted; the frozen-set claim in
   any doc needs the pin in-repo.
5. `results/ablations/olmoe_minflow_full.csv` sits untracked in the **archived
   FLAME-MoE repo** — wrong repo entirely; move to temporal-moe or the archive dir.

Runs (only one blocks completeness):
- **Trained controlled pair under preserve** ({0,1,15} vs {0,1,2} at matched memory,
  15M tokens each + downstream): ~2 GPU-hours. Without it, part 2 item 3 keeps the
  stated limit sentence; with it, the controlled-pair logic of the old section is
  restored on clean data. Recommended.
- Strategy bake-off re-run (7 arms × 50M): ~10 GPU-hours. Not recommended now; the
  capacity-cliff claim is not load-bearing for the current program.

## 6. Open questions, one sentence each

1. Run the 2-hour preserve controlled pair before the rewrite lands, or ship with the
   training-free joint table plus the {14,15} trained cell and the limit sentence? —
   **recommend run it**; it is the difference between "corrected" and "corrected with
   the same experimental design."
2. Mixed-era `layer_freeing_downstream.csv` `ce_free_*` rows: delete the era-trained
   cells, or keep them with re-based references and an era column? — **recommend
   delete from the live CSV, archive copy already exists**; the rewrite cites none.
3. `results/ablations/layer_freeing_RESULTS.md` carries era numbers with no banner —
   archive it beside its CSVs or banner it? — **recommend archive move** in the
   file-moves commit (decision 5 separation).
4. `layer_freeing_damage.png` disposition — **recommend move to
   `results/archive/olmoe_wrong_renorm/figures/`** with one line in its README, same
   commit as the other moves.
5. Where does the cross-model program's prose home stay — `sweep_RESULTS.md` as now,
   or promoted under `docs/research/`? — **recommend stay**, with a routing-index row;
   promotion is a rename decision that belongs to the file-moves commit if wanted.
