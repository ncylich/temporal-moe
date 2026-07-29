# Per-Layer Embeddings (PLE) for OLMoE residency adaptation

Implementation of `PLE_PLAN.md`. Phase 0 only: the module, the parity control, the accounting,
and the checks. No Phase 1 cell has been launched.

## Files

| file | what it is |
|---|---|
| `ple.py` | the factored PLE table and the post-MoE install/uninstall |
| `train_ple.py` | training cell; `--rank off` is the recipe-C parity control |
| `residency.py` | the rolling-residency mask, copied from the adaptation program (see below) |
| `olmoe_paths.py` | path contract for the base checkpoint and corpus, plus manifest verification |
| `accounting.py` | parameter / bandwidth table and corpus coverage |
| `parity_report.py` | collects the parity runs into `results/ablations/ple_parity.csv` |
| `placement_check.py` | proves the post-MoE placement leaves same-layer routing untouched |
| `grad_check.py` | proves the table receives gradient *through* gradient checkpointing |
| `zero_check.py` | the zero-property check, at init and after a cell trains |
| `row_norms.py` | per-cell diagnostic: mean row norm bucketed by occurrence count (no GPU) |

## Where the artifacts come from

`residency.py` is `olmoe_residency.py` from the adaptation program, copied here with two changes
only — the two hardcoded absolute paths replaced by resolution through `analysis/paths.py` — plus
one added function, `load_c_adapted`, which applies a saved arm-C delta. The residency scan itself
is untouched and still calls `temporal/temporal_router.py` verbatim.

The unmodified original is now archived at `scripts/adaptation/olmoe_residency.py`. The two files
are deliberately kept separate and are **not** interchangeable: the archived copy is the record of
what produced the published numbers and must stay byte-identical to what ran, while this one is the
working module. `diff scripts/adaptation/olmoe_residency.py analysis/ple/residency.py` shows the
whole delta.

The base checkpoint and the 4.4 GB corpus are resolved by `olmoe_paths.py`, in order:
`$TMOE_OLMOE_MODEL` / `$TMOE_OLMOE_DATA`, then `$TMOE_OLMOE_HOME`, then `<repo parent>/olmoe-adapt`.
The corpus files are listed in `results/MANIFEST.csv` with sha256; `olmoe_paths.py --full` checks a
local copy against it, which is what licenses using local disk instead of re-downloading.

## Design decisions that differ from the plan's literal text

**Gate initialization.** `PLE_PLAN.md` §2 asks for both a zero-initialized table and a gate
"initialized so the branch starts inert". Both at once is a permanent fixed point: with the table
at zero the contribution is already exactly zero, and a zero gate additionally zeroes the gradient
reaching the table, so `dL/dU = dL/dV = dL/dg = 0` and nothing ever moves. Measured directly:

```
gate=0.0: contribution_zero=True  dU_nonzero=False dV_nonzero=False dg_nonzero=False
gate=1.0: contribution_zero=True  dU_nonzero=True  dV_nonzero=False dg_nonzero=False
```

The gate is therefore initialized to 1.0. Inertness at step 0 comes from the zero table, which is
what actually gives bit-identical parity and the rare-row zero property; the gate is kept only
because §2 specifies a learned per-layer scale. `dV` and `dg` start at zero and become nonzero once
the table leaves zero, which is the intended order.

**Weight decay coefficient: 0 on every rung, decided.** §2 originally said to inherit the
coefficient "the C recipe already uses", which is an explicit `weight_decay=0.0` in all four
adaptation trainers (`train_bakeoff.py:51`, `train_router.py:31`, `train_fprime.py:85`,
`train_cal2.py:86`, now archived verbatim under `scripts/adaptation/`), so inheriting it literally
would disable the mechanism §2 argues for. That conflict was raised and resolved in favour of 0:

* the ladder exists to measure whether constraining **rank** denoises underdetermined rare-token
  rows. Regularizing by rank and by decay at once makes a null at low rank unattributable between
  the two. At 0 the ladder is single-axis and flag-off parity is exact by construction.
* §9's `λ*` is **not** a source for this coefficient. λ is pseudo-observations added to `n_t`, in
  count space, giving the `n_t/(n_t+λ)` shrinkage curve; decoupled weight decay is a loss
  coefficient, and against Adam's second-moment-normalized step a row's equilibrium norm goes
  roughly as frequency/wd — linear in frequency. The functional forms differ, so transferring `λ*`
  into the optimizer would manufacture rigour that is not there. λ stays in §9, which trains nothing.

The cost is acknowledged rather than hidden: with Adam and no decay, a row seen once takes a step
nearly the size of a row seen ten thousand times, because after one observation `v ≈ g²` and the
update is `≈ lr·sign(g)`. Full rank is therefore exposed to memorizing noise in rare rows. That is
the **pre-registered** reason full rank might lose, and losing that way is a result.

`--table-wd` stays a settable parameter group, at 0. `row_norms.py` is the diagnostic any non-zero
value should be set against; see its docstring for the stop-and-report trigger.

**Gradient clipping is per-surface.** The C parameters are clipped to norm 1.0 among themselves, as
in the reference trainer, and the PLE tensors are clipped separately. Clipping them jointly would
make the C-surface updates depend on the PLE gradient norm, so a rank-ladder cell would differ from
the flag-off control in the router and norm gains as well as in the table, confounding PLE's
contribution with a changed C trajectory.

## Placement

Post-MoE only, added to the layer output:

```
out = h + MoE(LN2(h)) + g_l * PLE[tok, l]
```

There is no pre-MoE flag and no code path that could produce one (§13). `placement_check.py`
verifies the guarantee that follows: layer 0's router logits are bitwise identical with and without
an active table, because PLE is added after that layer's MoE. Deeper layers' routing does move,
which is inherent to writing into the residual stream and is not the pre-MoE failure mode.

## Reproducing Phase 0

```bash
export TMOE_ROOT=$(git rev-parse --show-toplevel)
PY=<the adaptation venv python>          # torch 2.4.1+cu124, transformers 5.12.1, bitsandbytes

$PY analysis/ple/olmoe_paths.py --full           # verify the corpus against MANIFEST.csv
$PY analysis/ple/accounting.py --accounting      # -> results/ablations/ple_accounting.csv   (CPU)
$PY analysis/ple/accounting.py --coverage --model-for-loss C
                                                 # -> results/ablations/ple_coverage.csv     (GPU)
$PY analysis/ple/zero_check.py --init            # init-time zero property                   (CPU)
$PY analysis/ple/placement_check.py              # post-MoE placement guarantee              (GPU)
$PY analysis/ple/grad_check.py                   # gradient survives checkpointing           (GPU)
# parity: two flag-off runs plus the unmodified reference, then
$PY analysis/ple/parity_report.py                # -> results/ablations/ple_parity.csv
```

After the first Phase-1 cell, as §4 item 5 specifies:

```bash
$PY analysis/ple/zero_check.py --trained <ple_table_TAG.pt> --train-tokens 50000000
$PY analysis/ple/row_norms.py  --table   <ple_table_TAG.pt> --train-tokens 50000000
```

Both define "covered" from the prefix of the shuffled order the cell actually consumed, not from
the whole 1B corpus — a row can only have moved if the cell saw its token. Measured uncovered-row
counts, which is what the zero property is tested on: 3,023 rows at 10M tokens, 1,113 at 50M, 801
at 100M.
