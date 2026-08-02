# Mechanism — what rolling residency does to MoE routing

Five documents, numbered in reading order. Start at 01 unless you know what you are looking for.

| | | read it when |
|---|---|---|
| [`01-findings.md`](01-findings.md) | What we know, by claim | You want the results. **Start here.** No history, no test identifiers, nothing superseded |
| [`02-corrections.md`](02-corrections.md) | What changed against the published write-up | You are revising the paper, or a number here disagrees with one you remember |
| [`03-methods.md`](03-methods.md) | Every probe: metric, range, direction, controls, limits | You need to know exactly what a number measures, or to decode an identifier like `C8` or `N5` |
| [`04-coverage.md`](04-coverage.md) | Which script produces which CSV, over which runs and layers | You are reproducing something, or checking whether a cell exists |
| [`delexicalization.md`](delexicalization.md) | The narrative write-up | You want the argument in prose rather than as claims |

The chronological record — every defect found, how, and whether the conclusion drawn from it survived
— is [`05-notebook.md`](05-notebook.md). It is deliberately last on this path: it is a lab notebook, not a
result, and it is long.

[`archive/`](archive/) holds the superseded documents and the pre-correction write-up. Nothing there
should be cited.

## Conventions

- **Claims name their evidence.** Every number in 01 says which CSV it came from, and was read from
  that file rather than copied from an earlier document.
- **Superseded text is deleted, not annotated.** Corrections live in 02 and history lives in the
  notebook, so a reader is never asked to diff two versions of a paragraph.
- **Identifiers stay out of the findings.** The analysis scripts use about forty short labels
  (`A1`–`A11`, `C1`–`C10`, `N1`–`N9`, `T1`–`T4`, `e1`–`e8`). They are a private language and belong in
  03 and 04.
- **Generated files say so.** `04-coverage.md` is emitted by `analysis/coverage_table.py`; editing it
  by hand will be overwritten.
- **`scripts/reproduce.sh` is the gate.** It runs every documented command, checks the tree is
  unchanged afterwards, runs the CSV linter, and fails on any warning not on an explicit allowlist.
  Run it before pushing.
