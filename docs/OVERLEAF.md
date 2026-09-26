# Agent notes: the paper and Overleaf

`paper/` is **a git submodule, not a normal directory**. It points at the Overleaf
project that hosts the ICLR 2027 submission. Its `origin` is Overleaf, its branch is
`main`, and inside it git behaves completely normally.

The old NeurIPS technical report lives in `LEGACY_paper/` and is a plain tracked
directory. It is superseded — do not edit it, and do not confuse the two.

## First time in a fresh clone

```bash
git submodule update --init paper
```

This needs Overleaf credentials (username `git`, password = an Overleaf Git
authentication token from Account Settings → Git Integration). Machines without them
— pods, CI — cannot check out `paper/`; skip it there and work on the rest of the repo.

## Pull down edits made in the Overleaf web editor

```bash
cd paper && git pull
```

## Make edits and push them

Ordinary git, because the submodule's root *is* the Overleaf project root:

```bash
cd paper
# edit main.tex, add figures, etc.
git add -A && git commit -m "describe the change" && git push
```

The change appears in the Overleaf editor as soon as the push lands.

## Record the new state in the parent repo

The monorepo tracks *which commit* of the paper it points at, so after pushing:

```bash
cd .. && git add paper && git commit -m "paper: bump to latest Overleaf state"
```

Skip this and the parent repo keeps pointing at the older paper commit.

## Constraints that will bite you

- **Overleaf rejects force pushes.** Stack commits on the current head. If you have
  diverged, `git pull --rebase` inside `paper/` rather than trying to overwrite.
- **The branch is `main`.** Overleaf's bridge rejects pushes to any other branch, and
  will not let you create new ones.
- **Do not run `git subtree` against Overleaf.** The monorepo history and the Overleaf
  history are unrelated by design; subtree push/pull will not work.
- Figures are committed into `paper/figures/`. Regenerating them means committing
  inside the submodule, not the parent.

## Building locally

```bash
cd paper && latexmk -pdf main.tex
```

Requires the `helvetic` and `courier` font packages on top of a minimal TeX Live; a
full TeX Live / MacTeX install has them already. Overleaf compiles it regardless.
