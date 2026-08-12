# Per-Layer Embeddings (PLE) — results

Program of record: [`PLE_PLAN.md`](../../PLE_PLAN.md). All numbers below are in
`ple_results.csv` (tidy, sliceable by its `group` column); code is in
[`analysis/residency/`](../../analysis/residency/README.md).

**Scope.** This covers per-layer embeddings only: PLE *adds* a token-indexed lookup while leaving
the rolling-residency constraint intact. The separate experiment that *relaxes* the constraint on
chosen layers is written up in [`layer_freeing_RESULTS.md`](layer_freeing_RESULTS.md) with its own
table, and is not mixed in here — the two share only a base model, an eval slice and a set of
published references.

**Metric.** BPB = cross-entropy nats ÷ 3.10891 on the Stage-1 audited held-out slice (dolmino
dclm), the divisor byte-derived as `ln2 × bytes_per_token`. **Lower is better.**
`recovery = 1 − (BPB − 0.6727)/(2.7507 − 0.6727)`: 0% is rolling residency `R=k=8` of 64 imposed
untrained, 100% is the base model with free routing. **2σ = 0.012 BPB; anything smaller is noise.**

---

## 1. Headline

**PLE is a token-efficiency win over the C recipe, not a new mechanism.** A rank-512 per-layer
embedding table co-trained with router and norm gains reaches **0.848854 at 50M tokens**, which ties
C@250M = 0.8505 (Δ 0.14σ). Same quality for a fifth of the tokens, at 42.5M parameters and 1 KB of
flash traffic per token — and r=128 does it at 10.6M parameters and 256 B/token.

PLE does not, however, reach the constraint price: the best PLE cell on the CE surface
(0.832730) ties LoRA alone and remains 2.2 points of recovery short of F′ = 0.8106.

## 2. What failed, and how conclusively

| axis | result | evidence |
|---|---|---|
| **PLE's mechanism** (§1's premise) | **refuted** | locus unmoved: 0.0932 no-PLE → 0.0964 with PLE, spread 0.0031, *opposite* to the pre-registered direction |
| **PLE stacked on LoRA** | no benefit | 0.8327 vs 0.8269 LoRA-alone; 0.49σ, wrong-side, at both ranks |
| **Calibrated initialisation** | no benefit, then harm | three arms tie (0.47σ, 0.08σ, 0.59σ); the strongest init is **1.23σ worse** |
| **Sequential vs joint** (§7) | null | 0.843163 vs 0.848854, 0.47σ |
| **Rank** | not binding above 128 | full/512/128 mutually within 2σ; r=32 real-worse by 1.31σ |

**The mechanism failure is the consequential one.** §1 argued PLE helps *because* it restores
token-specific information the constraint strips out; §8.1 pre-registered that token AUC should rise
and context-minus-token move toward zero. Measured with one probe on one dataset across three
models, adding PLE moves the locus by 0.003 in the **wrong** direction. §8's own clause then
applies: the gain is generic capacity, not lexical restoration, and the paper claim must be
weakened. That conclusion depends on the **no-PLE control** — this probe reports 0.0932 on the C
surface where the published `hf_delex.py` reported 0.0493 for CE-adapted, so cross-probe comparison
would have manufactured a spurious "PLE moved the locus from 0.049 to 0.095".

## 3. Trained cells

| cell | BPB | recovery | config |
|---|---|---|---|
| ce_ple_128 | 0.832730 | 92.30% | CE + PLE r128 |
| ce_ple_512 | 0.833799 | 92.25% | CE + PLE r512 |
| seq_ple_512 | 0.843163 | 91.80% | PLE introduced at 50M of 100M |
| **ladder_r512** | **0.848854** | **91.52%** | PLE r512, 50M — ties C@250M |
| cal_seq_512 | 0.850227 | 91.46% | calibrated init at 50M |
| ladder_r128 | 0.854388 | 91.26% | PLE r128, 50M |
| cal_r512 | 0.854536 | 91.25% | calibrated init |
| cal_full | 0.857931 | 91.09% | calibrated init |
| ladder_full | 0.858867 | 91.04% | PLE full rank, 1.65B params |
| ladder_r32 | 0.864538 | 90.77% | PLE r32 |
| calstack_full | 0.873675 | 90.33% | init from the 53.13% training-free stack |

References: base free 0.6727 · impose 2.7507 · C@50M 0.8791 · C@250M 0.8505 · CE@50M 0.8269 ·
F′ 0.8106.

## 4. Per-layer residency damage

Moved: see [`layer_freeing_RESULTS.md`](layer_freeing_RESULTS.md). It is a property of the
constraint, not of PLE, and belongs with the relaxation experiments.

## 5. Training-free adaptation, and why order matters

Closed-form corrections with **zero gradient steps**. Stage 1 independently replicates Cal-0
(31.48% vs published 31.5%).

| configuration | BPB | recovery |
|---|---|---|
| imposed, no correction | 2.750704 | 0% |
| Cal-0 calibrated norms alone | 2.096512 | 31.48% |
| calibrated PLE alone (full rank) | 1.973545 | 37.40% |
| norms **then** PLE | 2.844982 | **−4.54%** |
| norms then PLE, same-scale capture | 3.236982 | **−23.40%** |
| **PLE then norms** | **1.646688** | **53.13%** |

**Reversing the order turns a catastrophe into the best training-free result in the program.** Norm
calibration is a *rescaling*; PLE is an *additive offset fitted against a fixed reference*. A
rescaling can be fitted last, absorbing whatever the previous correction left. An additive offset
cannot: once norms have moved the frame, the offset is estimated in one scale and applied in
another, and the two double-count.

Note Cal-0 calibrated **RMSNorm gains only** — its docstring says "base router + matched norms". No
closed-form router calibration exists here, and residency damages the router by restricting *which
experts are eligible*, a selection effect no rescaling addresses.

**Base free routing is the correct capture target**, established on two surfaces: same-surface
capture scored 2.2142 vs 1.6399 on the trained 50M surface, and −23.40% vs −4.54% on the calibrated
norms. And the 53.13% state is a *bad* place to start training — `calstack_full` finished 1.23σ
**worse** than zero-init, because its norms begin displaced from base and training spends capacity
walking them back. Cheap inference model, poor warm start; the two uses are in tension.

## 6. Verification

| check | result |
|---|---|
| flag-off parity vs the unmodified reference trainer | **bitwise identical** forward *and* backward, all quantities 0.000e+00, once Flash Attention's non-deterministic backward is disabled |
| recipe C replication | 0.877859 vs published 0.8791 at 50M |
| impose / Cal-0 replication | 2.750704 vs 2.7507; 31.48% vs 31.5% |
| zero property | 160/160 held-out rows bit-zero, **all 160 covered in training** — rows eligible to move that didn't |
| gradient through checkpointing | identical loss on/off, table gradient present in both, hitting exactly the batch token ids |
| post-MoE placement | layer-0 router logits bitwise identical with an active table (`1000000000000000`) |
| row norms vs frequency | monotone rising, 0.282 at one occurrence to 1.449 above 100k — no rare-row blow-up, so `wd=0` holds |
| memory | activations dominate: 48.2 of 61.1 GiB at mb16; full-rank table adds 15.35 GiB |

**Parity was resolved as a code question, not a statistical one.** Replicates could only ever bound
the difference; a deterministic forward/backward comparison proved the implementations identical and
localised the reference's 0.002698 self-spread to Flash Attention's backward. Flash stays **on** for
every training cell (§10) — the ~1e-3 relative gradient noise is below seed variance and not worth
trading throughput for.

**σ = 0.006 measures the wrong thing.** `eval_noise_sigma.py` scores the *base* model on *disjoint*
subsamples, i.e. data-slice noise. Every arm is scored on the same fixed 256-pack subset and the
eval forward is bitwise deterministic, so subsample variance contributes nothing to inter-arm
differences — the only source is training nondeterminism (~0.0024 from |B1−B2|). The 2σ = 0.012 bar
is therefore conservative by roughly 2.4×, and was retained unchanged.

## 7. Not done

- **§8.2 downstream 10-task evaluation.** `lm_eval` is incompatible with transformers 5.12.1
  (`AutoModelForVision2Seq` removed). Not run, not faked. lambada was billed as the sharp test of the
  lexical claim, though §8.1 has already answered that question.
- **250M depth run.** Gated on a CE cell below 0.8149; best was 0.832730, so it never fired — the
  plan's own arithmetic, saving 3.6 GPU-hours on a configuration that ties an existing result.
- **`{0,1,2}` trained cell**, cancelled. See §4.

## 8. Recommendation

Write PLE up as an **efficiency and serving-cost** result versus recipe C — 5× token efficiency at
256 B/token — and drop the lexical-restoration framing, which the locus probe does not support. PLE
is not an additional mechanism on top of adapter capacity.

For the constraint-relaxation frontier and the 95% question, see
[`layer_freeing_RESULTS.md`](layer_freeing_RESULTS.md).

§13's follow-on (DeepSeek Engram, richer lookups) is now less attractive: the axis produced a real
efficiency win but no evidence for the mechanism that motivated it, and it does not compose with
LoRA.
