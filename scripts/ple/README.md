# `scripts/ple/` — run scripts for the layer-freeing and attention rounds

Two kinds of thing live here and they should not be confused.

## Reusable

| script | what it does |
|---|---|
| `snapshot_cells.sh` | copies the authoritative per-cell result JSONs into `results/ablations/cells/` and fails if the snapshot has gone stale or any file will not parse. Run after any new cell. |
| `watchdog.sh` | detached hourly heartbeat into `results/ablations/overnight_heartbeat.log`, committed and pushed. Not a child of any agent process, so it survives when monitors do not. Never touches the GPU. |

## One-shot, kept as the record of what ran

These drove specific rounds on 2026-08-02/03. They are idempotent — each skips any cell that already
has a result — so re-running one is safe, but none of them is a general-purpose entry point. They are
kept rather than deleted because the commit history refers to them by name and because what was run,
in what order, and under what deadline is part of the result.

| script | round | outcome |
|---|---|---|
| `overnight_chain.sh` | `{0,1,2}`, `{0,1,14,15}`, the `{0,1,15}` replicate, then the winner to 200M and downstream | completed |
| `extend_250M.sh` | both free-set cells to 250M, matching the published comparators, plus downstream | completed |
| `attn_lora_cell.sh` | first attention cell, with its downstream gated on beating the control | completed, gate passed |
| `attn_250M.sh` | attention to 250M | killed by the disk quota; see below |
| `recover_attn250M.sh` | recomputed the final 10M from the intact 240M checkpoint | completed |
| `autonomous_queue.sh` | four cells against an 18:00 UTC deadline | **retired mid-run.** Its stage B hit CUDA OOM and its stage D would have taken the slot the replicate needed. Superseded by `retry_attn_cells.sh`. |
| `retry_attn_cells.sh` | re-ran the OOMed replicate and scored the cell the retired queue left unscored | completed |

## Two failures worth carrying forward

**The volume enforces a quota that `df` does not report.** `df` read 65% used with 240 TB free while
a plain `dd` returned `Disk quota exceeded`. A checkpoint write hitting it killed the trainer
mid-file — twice, truncating at exactly 469762048 bytes both times — with no traceback, and took the
driver script with it. Free space in `df` is not evidence that a write will succeed, and "no
traceback" is not evidence of an external cause. `autonomous_queue.sh` now prunes intermediate
checkpoints before each cell; only a cell's last checkpoint is needed, since every number it produced
is in its result JSON.

**The attention cells sit ~0.2 GiB inside the memory ceiling at `--mb 16`.** The same configuration
cleared it twice and then failed once, so it is fragmentation rather than a fixed limit. Run them
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and fall back to `--mb 8 --accum 2` — which
holds the effective batch at 16 and is therefore the same cell, not a cheaper one.
