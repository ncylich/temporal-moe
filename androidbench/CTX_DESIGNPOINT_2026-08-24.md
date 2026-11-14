# Pixel 10a: design-point turnover across context depth (2026-08-24)

Answers the two open Pixel items in `paper/TODO.md`: re-measure with **prescribed
design-point turnover** instead of the degenerate random-weight router, against a
**directly measured** ceiling instead of one inferred from byte accounting.

Device: Pixel 10a "stallion" (Tensor G4, 7.75 GB RAM, UFS 3.1, Magisk root), same rig as
the 2026-07-24 campaign. Governors pinned `performance`, storage queue `scheduler=none
nomerges=2 iostats=0`, `-t 4`, n=128 decode tokens x 3 reps per arm.

Raw data: `results/ctx_designpoint_2026-08-24.csv`. Runner: `run_ctx_designpoint.py`.

---

## Headline

At a **10.7x expert-memory cut** (resident set = the active experts only, R=k), with
prescribed one-swap-per-layer-per-token turnover:

| ctx depth | fine (E=192, k=18) | % of ceiling | k24 (E=256, k=24) | % of ceiling |
|---|---|---|---|---|
| 0    | 20.70 tok/s | 63.9% | 20.44 tok/s | 65.5% |
| 1024 | 17.80 | 70.0% | 17.61 | 71.4% |
| 2048 | 15.66 | 77.4% | 15.21 | 74.9% |
| 4096 | 12.16 | **83.4%** | 12.17 | **88.0%** |

**Definitions** (so this table stands alone):

- *decode tok/s* — generation throughput, higher is better.
- *ceiling* — the same expert shape (same active-expert count k, same per-expert width)
  with the **largest total expert count E that fits fully resident on the device**, run
  fully resident with the streaming machinery off (`ceilplain`), measured **at the same
  context depth** as the arm it divides. Per `BASELINE_POLICY.md`. Ceilings used here:
  fine-shape `e80` = 32.42 / 25.44 / 20.22 / 14.58 tok/s and k24-shape `e100n` = 31.21 /
  24.66 / 20.31 / 13.83 tok/s at ctx 0 / 1024 / 2048 / 4096.
- *% of ceiling* — decode ÷ same-depth ceiling. Higher is better; 100% = no measurable
  cost from streaming experts off flash.
- *ctx depth* — tokens of KV-cache context established before the timed decode window
  (`llama-bench -d`). **`-d` defaults to 0**, which is what the ad-hoc scripts behind the
  2026-07-24 curve used.
- *10.7x cut* — E/R: 192/18 for fine, 256/24 for k24. Expert memory only.

## Findings

1. **Context depth is the strongest lever measured.** Both shapes gain ~20 points of
   ceiling-fraction from ctx=0 to ctx=4096, monotonically. Mechanism: attention cost grows
   with depth while the enforced per-layer swap cost is fixed, so swap overhead amortizes
   against a larger per-token budget. This is why a phone benchmark run at depth 0
   understates the technique for any realistic chat/agent workload.

2. **Fine-graining to k=24 is mildly favourable and moves far fewer bytes.** k24 fetches
   8.5-8.8 GiB per arm vs fine's 11.2-11.3 GiB (**~24% fewer bytes**) at equal active
   compute (k*ff invariant: 18x384 = 24x288 = 6912). It is level with fine at short
   context and pulls ahead at 4096 (88.0% vs 83.4%). Consistent with S3-35's earlier
   narrow-reshape result (+2.4%): per-request fixed cost dominates bytes on this storage,
   so byte reductions convert to wall-time at a heavy discount.

3. **R above k is inert in this regime, by construction.** Control arm: R=36 measured
   20.51 tok/s vs R=18's 20.70, with **byte-identical** fetch and eviction counts. Cause is
   structural, not noise — see pitfall #26. An honest R-curve needs the other regime
   (`run_rcurve_norepack.py`), where R=18 -> 12.04 and R=36 -> 17.29 tok/s (+44%) as slack
   lets jittered-out experts survive instead of being refetched
   (`results/rcurve_norepack_2026-08-24.csv`).

## What this is NOT

- **Not a replication of the 2026-07-24 curve.** That curve (R=18 -> 5.65 tok/s, R=36 ->
  10.49) came from the *natural* random-weight router churning 2.09 experts/layer/token
  with no two-pass overlap. This run enforces ~1.05 experts/layer/token. Neither regime
  tried here reproduced those absolute numbers (TWOPASS 20.70, natural-jitter 12.04 at
  R=18), so **do not present these as the same measurement re-run**.
- **Not a trained router.** `LLAMA_TEMPORAL_TWOPASS` *prescribes* one swap per layer per
  token, which is the documented design point of a temporally-coherent trained router
  (LEDGER S3-9). It is an emulation of that design point, and every claim built on these
  numbers must say so. Measuring an actually-trained temporal router remains open.

## Validity gates that passed on every reported arm

Real turnover (~51,800 evictions/arm, 8.5-11.3 GiB genuinely read from flash), zero swap,
sustained cool-gate before each arm, and residency-weighted mean clock >= 97% of rated
across the arm's actual runtime. Throttled arms were voided and retried (ctx=4096 needed
up to 3 attempts). Voided attempts are retained in the raw CSVs with `void=True`.
