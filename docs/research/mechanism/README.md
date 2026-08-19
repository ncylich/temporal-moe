# Mechanism, what rolling residency does to MoE routing

Five documents, numbered in reading order. (01 was rewritten 2026-08-19 from the
data layer; the deprecated version and the probe-era content live in `archive/`.)

| | | read it when |
|---|---|---|
| [`01-findings.md`](01-findings.md) | What we know, by claim | Rewritten 2026-08-19 from committed sources: constraint costs, thinking/length, fluency, adaptation, measurement corrections |
| [`02-corrections.md`](02-corrections.md) | What changed against the published write-up | You are revising the paper, or a number here disagrees with one you remember |
| [`03-methods.md`](03-methods.md) | Every probe: metric, range, direction, controls, limits | You need to know exactly what a number measures, or to decode an identifier like `C8` or `N5` |
| [`04-coverage.md`](04-coverage.md) | Which script produces which CSV, over which runs and layers | You are reproducing something, or checking whether a cell exists |
| [`delexicalization.md`](delexicalization.md) | The narrative write-up | You want the argument in prose rather than as claims |

The chronological record, every defect found, how, and whether the conclusion drawn from it survived, is [`05-notebook.md`](05-notebook.md). It is deliberately last on this path: it is a lab notebook, not a
result, and it is long.

[`archive/`](archive/) holds the superseded documents and the pre-correction write-up. Nothing there
should be cited.

## Finding the layer-lexicality work

It ran as its own program for weeks and has no directory, deliberately. Its results are a *depth cut*
of three different quantities, so they live with the quantities they cut:

| the depth result | lives in |
|---|---|
| The contextual share of routing rises with depth, clearly unconstrained and weakly constrained, so the regime gap narrows | `archive/01-findings-deprecated-20260814.md` §1 (probe era; uncorrected) |
| Routing demand becomes more cacheable with depth | `archive/01-findings-deprecated-20260814.md` §1 (probe era; uncorrected) |
| The last MoE layer costs the most to constrain, in seven of seven measurements; the first in two | `archive/01-findings-deprecated-20260814.md` §4 (probe era; uncorrected) |
| Single-layer damage does not predict which layers to free | `archive/01-findings-deprecated-20260814.md` §5 (probe era; uncorrected) |
| Its hypotheses, what falsified them, and what was withdrawn | `05-notebook.md`; retractions in `02-corrections.md` §4 |
| Its superseded plan documents | `archive/LAYER_LEXICALITY*.md` |

Putting it in one directory would mean either splitting the base results it cuts, or holding a second
copy of them. This repository's every documented drift came from a claim having two homes.

## Conventions

- **Claims name their evidence.** Every number in 01 says which CSV it came from, and was read from
  that file rather than copied from an earlier document.
- **Superseded text is deleted, not annotated.** Corrections live in 02 and history lives in the
  notebook, so a reader is never asked to diff two versions of a paragraph.
- **Identifiers stay out of the findings.** The analysis scripts use about forty short labels
  (`A1` to `A11`, `C1` to `C10`, `N1` to `N9`, `T1` to `T4`, `e1` to `e8`). They are a private language and belong in
  03 and 04.
- **Generated files say so.** `04-coverage.md` is emitted by `analysis/coverage_table.py`; editing it
  by hand will be overwritten.
- **`scripts/reproduce.sh` is the gate.** It runs every documented command, checks the tree is
  unchanged afterwards, runs the CSV linter, and fails on any warning not on an explicit allowlist.
  Run it before pushing.

## Working rules

Each of these comes from a mistake made here, recorded in `05-notebook.md`, rather than from general
good practice. They apply to anyone working in this directory, human or agent.

**Enumerate from the data, not from the code.** Build an inventory of results by walking
`results/ablations/`, never by reading the analysis scripts and listing what they write. About one
committed result file in eight has no producer script anywhere in the repository, because it predates
the tooling or the producer was left on a pod. A code-first inventory is blind to those by
construction, and that is how a sixteen-layer per-layer sweep stayed unread while this document set
argued from a weaker version of the same result. A result file with no producer is a defect to report,
not a reason to skip it.

**Every file in scope gets a written verdict.** Either a claim it supports with the number, or a line
saying what it measures and why it is not load-bearing, or "I could not tell what this measures",
which is acceptable. Silence is not, because silence is indistinguishable from the file not existing.
The failure this prevents is selection by ease of interpretation: of ten replay analyses here, the six
with self-explanatory column names were written up, three that needed interpretation were opened and
dropped, and two never made the list. That selects against the harder results, which are not the less
important ones.

**Recompute before restating.** A claim is recomputed from its source file before it is repeated, not
copied from an earlier document. Every substantive error in these documents was caught by recomputing
and none by rereading.

**Fix the class, not the instance.** A defect someone points at is a sample. Find every instance of
the same kind, fix the set, and say in the commit message how many there were and how you searched.
Three values here were corrected in one place and left stale in their twin.

**Grep for the old value before committing a corrected one.** The mechanical form of the rule above.

**Warnings are failures.** A run that prints warnings is not clean. Two defects here announced
themselves on every invocation and went unread for weeks.

**Idempotent, faithful and sane are three checks, not one.** Two runs agreeing says nothing about
whether the output matches what is committed, and neither says anything about whether the numbers
still mean something. A fix here once produced twelve zero-width confidence intervals that were
perfectly idempotent.
