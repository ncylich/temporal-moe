# Archive — superseded mechanism documents

These are kept for provenance, not for reading. Everything true in them has been rewritten into the
numbered documents one level up, from the committed CSVs rather than by copying. Nothing here should
be cited; where a number here disagrees with `01-findings.md`, the findings document is correct.

Two reasons a document lands here:

1. **Its numbers cannot be regenerated** — the runs they were computed on are absent from
   `MANIFEST.csv` and from disk.
2. **It was organised by when the work happened** rather than by what is true, so its structure
   preserves every superseded claim alongside its correction.

| file | why archived | where its content went |
|---|---|---|
| `LAYER_LEXICALITY.md` | Reason 2. Held the round-1 verdict on the per-layer hypotheses, superseded twice, plus the T1 write-up | Findings §2, §3; the T1 result in §3.2; method text in `03-methods.md` |
| `LAYER_LEXICALITY_ROUND2.md` | Reason 2. Held the current verdict, but reachable only by knowing it superseded the file above | Findings §2–§4; the per-layer cost story in §3.2 |
| `MECHINTERP_RERUN_PLAN.md` | Reason 2. A housekeeping plan that accumulated the corrections register in its §7 | §7 became `02-corrections.md`; the coverage table became `04-coverage.md` |
| `probe-results.md` | **Reason 1.** Its own banner records that every per-model number came from runs no longer on disk | Superseded by the replay re-run over 22 preserved logs; nothing carried forward |
| `probe-replay-e1-e8.md` | **Reason 1.** Same provenance failure — the five runs behind E1–E8 are absent everywhere | Same; the current replay numbers are in the `e1`–`e8` CSVs and `04-coverage.md` |
| `delexicalization-original.md` | The paper-facing write-up as published, before corrections were applied | The corrected version is `../delexicalization.md`; the delta is `02-corrections.md` |

**`TODO.md` is deliberately still at the repository root.** It is the process record and the
reproduction gate depends on it — `analysis/todo_status.py` parses it and `scripts/reproduce.sh` runs
that. Migrating it to `05-notebook.md` requires updating both, and is tracked as follow-on work rather
than done here, because filing should not break a working gate.
