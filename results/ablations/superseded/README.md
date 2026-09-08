# Superseded results

Files here are kept as a record and must not be cited for current claims. A file earns a place in this
directory only when a replacement covers **every run it contains**, so that moving it loses nothing. A
file that is the sole record of even one run stays in `results/ablations/`, however old its method, and
carries its verdict in that directory's index instead.

That test is stricter than it sounds, and it is why this directory holds one file rather than the
half-dozen an earlier pass proposed for it. The structural and specialization tables look superseded
and are not: between them they are the only surviving measurement of nine runs, four of whose
checkpoints no longer exist anywhere.

| file | superseded by | verdict |
|---|---|---|
| `mechinterp_logitlens.csv` | `mechinterp_lens.csv` | Both its runs appear in the replacement. Its layer numbers run 1–3 where the replacement runs 2–4, and that offset is the defect itself: the capture filed router logits under a 1-based layer number and expert output vectors under a 0-based module index, so every output vector was attributed one layer too shallow and the deepest layer got none. The output-side claims computed from this file were retracted. |

The retraction is recorded in `docs/research/mechanism/02-corrections.md`, where this file is the
evidence behind defect A. That is the reason to keep it rather than delete it: a correction whose
evidence has been thrown away cannot be checked.


## instruct_genbench_vllm_history.csv (added 2026-08-14)
Every superseded, probe, and invalid-class row partitioned out of the live instruct
benchmark CSV (original order preserved; no header row — column layout matches the
live file). Replacement coverage: every non-probe run it contains is superseded by a
single-pass row in the live file; probe rows (smoke_*, lfm25_vllm) have no replacement
and are retained purely as history. Validity rules: analysis/residency/partition_eras.py; protocol: ../DATA_CONTRACT.md.
