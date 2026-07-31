#!/usr/bin/env python3
"""The single run registry every mechinterp script reads, replacing the per-function constants
(`HEADLINERS`, `ALL_TEMPORAL`, `CELLS`, `PAIRS`) that this audit exists to remove.

A run's identity comes from two sources on disk, never from a list in a script:

  results/MANIFEST.csv   which runs exist and which artifacts each preserved
  <run>/run.meta         the architecture that run was trained with

`run.meta` has two formats in the wild. Every `experiments/run.sh` run writes a two-line form with
`shape= H= L= grain= num_experts= topk= temporal= flops=`; the flame38m 1e18 runs write a single
line with `mode=temporal grain= num_experts= topk=` and no `L=` or `flops=`. Both are parsed here so
callers never see the difference. Budget for the single-line form is recovered from the run name.

Layer counts deliberately are NOT taken from run.meta. `L` there is the transformer depth, and MoE
layers are a subset of it (layer 1 is a dense FFN in every config, `--moe-layer-freq
"[0]*1+[1]*(L-1)"`). The authority on which MoE layers a measurement can cover is the artifact
itself: the keys of the capture's `layers` dict, or of the router log's. `moe_layers()` reads them
from there, which is what stops a hardcoded range from silently dropping the deep half of a model.
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, RUNS

__all__ = ["Run", "runs", "get", "moe_layers", "budget_of", "depth_of", "shape_of",
           "selection"]

MANIFEST = os.path.join(ROOT, "results/MANIFEST.csv")


class Run:
    """One trained run, with the artifacts it preserved and the architecture it used."""

    def __init__(self, name, files):
        self.name = name
        self.files = files                      # basenames preserved for this run
        self.meta = _parse_meta(name)
        self.grain = self.meta.get("grain")
        self.E = self.meta.get("num_experts")
        self.k = self.meta.get("topk")
        self.temporal = self.meta.get("temporal")
        self.dense = self.meta.get("dense")     # dense-FFN control: no experts, no router at all
        self.budget = self.meta.get("flops") or budget_of(name)
        self.depth = self.meta.get("L")         # transformer depth, or None for the 1e18 form

    # ---- artifact predicates: what analyses this run can actually feed ----
    # An artifact counts if MANIFEST.csv preserved it OR it exists on disk. The second clause is not
    # redundant: a capture produced locally by the Step 3 sweep is not in the manifest, and without it
    # every downstream analysis silently skipped freshly captured runs while reporting success.
    def _has(self, fname):
        # Size > 0: a failed capture can leave a zero-byte file behind, which then reports as present
        # and makes every downstream analysis crash on an empty layers dict rather than skip the run.
        p = self.path(fname)
        return fname in self.files or (os.path.exists(p) and os.path.getsize(p) > 0)

    @property
    def has_router_log(self):
        return self._has("router_log.pt")

    @property
    def has_capture(self):
        return self._has("delex_capture.pt")

    @property
    def has_ckpt(self):
        return self._has("common.pt") or os.path.isdir(os.path.join(RUNS, self.name, "ckpt"))

    def path(self, fname):
        return os.path.join(RUNS, self.name, fname)

    @property
    def regime(self):
        """'temporal' (rolling residency), 'full' (unconstrained MoE), or 'dense' (no experts).

        The dense control has no router, so every routing metric here is undefined for it; it is the
        isoFLOP quality floor, not a third routing regime.
        """
        if self.dense:
            return "dense"
        return "temporal" if self.temporal else "full"

    @property
    def grain_label(self):
        """Human-readable granularity, e.g. 'fine 18/192'. Matches the published figure wording."""
        if self.E is None or self.k is None:
            return "unknown"
        return f"{'fine' if self.grain == 3 else 'coarse'} {self.k}/{self.E}"

    def label(self):
        """Self-describing label for a CSV row: regime, granularity, budget."""
        return f"{self.regime} {self.grain_label} @{self.budget}"

    def __repr__(self):
        return f"Run({self.name}, {self.regime}, {self.grain_label}, {self.budget})"


# Transformer depth by run.sh shape name. Values for shapes whose runs preserved a run.meta are read
# from it; the rest are read from a sibling run in MANIFEST.csv that shares the shape, so nothing here
# is guessed. Note s19opt is 14, not the 9 that both mechanism plan documents assumed.
SHAPE_DEPTH = {"s0": 4, "s1": 5, "s2": 6, "s3": 7, "s19opt": 14,
               # the 1e18 panel is L=9 at every hidden size, verified against the checkpoints
               "s38m": 9, "s192f": 9, "s512f": 9}

# The 1e18 panel was trained by its own launchers in experiments/scale_1e18_1e19/ with the geometry
# hardcoded, so those runs' run.meta records ffn/num_experts/topk but no shape= and no flops=. These
# are the run.sh shape names added for them; each was verified to derive every field of the
# corresponding run.meta exactly (see the case block in experiments/run.sh).
NAME_SHAPE = (("flame38m", "s38m"), ("flame192", "s192f"), ("flame512", "s512f"))

# The 1e19 runs carry shape=s19opt in run.meta, so they resolve on a machine that has the artifact
# tree — and only there. In a clone without it there is no run.meta, no NAME_SHAPE prefix matches
# (their names begin with moe_/g1_/temporal_/dense_), and the token fallback finds no shape name
# inside `moe_coarse_1e19`. depth_of() then returns None, plot_locus_by_layer drops all three 1e19
# series, and the figure quietly loses a third of its curves. That is why this looked resolved from
# the pod for three rounds while being broken for anyone else.
#
# A suffix rule rather than three exact names, because it also covers dense_1e19. Verified against the
# checkpoints, not the docs: all four runs' run.meta record `shape=s19opt H=800 L=14`, matching
# SHAPE_DEPTH["s19opt"] = 14. No other run in the registry ends in _1e19.
SUFFIX_SHAPE = (("_1e19", "s19opt"),)


def shape_of(name):
    """run.sh shape name for a run: from run.meta if it records one, else from the run-name prefix."""
    sh = get(name).meta.get("shape")
    if sh:
        return sh
    for prefix, shape in NAME_SHAPE:
        if name.startswith(prefix):
            return shape
    for suffix, shape in SUFFIX_SHAPE:
        if name.endswith(suffix):
            return shape
    for shape in SHAPE_DEPTH:
        if re.search(rf"(^|_){shape}(_|$)", name):
            return shape
    return None


def depth_of(name):
    """Transformer depth for a run, from its own run.meta if present, else from its shape."""
    r = get(name)
    if r.depth:
        return r.depth
    return SHAPE_DEPTH.get(shape_of(name) or "")


def budget_of(name):
    """Compute budget from the run name, for runs whose run.meta omits flops= (the 1e18 form)."""
    m = re.search(r"(1e1[6-9])", name)
    if m:
        return m.group(1)
    # The 1e18 isoFLOP panel: flame38m is the middle, flame192/flame512 the left/right flanks.
    if name.startswith(("flame38m", "flame192", "flame512")):
        return "1e18"
    return "unknown"


def _parse_meta(name):
    """<run>/run.meta -> dict. Tolerates both formats and a missing file."""
    p = os.path.join(RUNS, name, "run.meta")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as f:
        text = f.read()
    for key, val in re.findall(r"(\w+)=([^\s]+)", text):
        out[key] = val
    for key in ("L", "grain", "num_experts", "topk", "H"):
        if key in out:
            try:
                out[key] = int(out[key])
            except ValueError:
                del out[key]
    if "temporal" in out:
        out["temporal"] = out["temporal"] == "1"
    elif "mode" in out:
        out["temporal"] = out["mode"] == "temporal"     # the flame38m single-line form
    if "dense" in out:
        out["dense"] = out["dense"] == "1"
    elif "mode" in out:
        out["dense"] = out["mode"] == "dense"
    return out


def _manifest_runs():
    """run name -> set of preserved artifact basenames, from MANIFEST.csv."""
    if not os.path.exists(MANIFEST):
        sys.exit(f"manifest not found: {MANIFEST}")
    out = {}
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            name = r["run_name"]
            if name.startswith("_"):             # _batch_logs, _lmeval_scratch: not runs
                continue
            out.setdefault(name, set()).add(os.path.basename(r["hf_path"]))
    return out


def runs(router_log=False, capture=False, ckpt=False, temporal=None, budget=None, on_disk=False,
         dense=None):
    """Every run matching the filters, in name order. All filters default to off.

    router_log/capture/ckpt: require that artifact be preserved (per MANIFEST.csv).
    temporal: True for residency-constrained runs, False for unconstrained baselines.
    dense:    True for the dense-FFN isoFLOP floor, False to exclude it (it has no router).
    budget:   '1e16'..'1e19', or a collection of them.
    on_disk:  additionally require the artifact be present locally, not merely preserved.
    """
    out = []
    for name, files in sorted(_manifest_runs().items()):
        r = Run(name, files)
        if router_log and not r.has_router_log:
            continue
        if capture and not r.has_capture:
            continue
        if ckpt and not r.has_ckpt:
            continue
        if temporal is not None and r.temporal is not temporal:
            continue
        if dense is not None and bool(r.dense) is not dense:
            continue
        if budget is not None:
            want = {budget} if isinstance(budget, str) else set(budget)
            if r.budget not in want:
                continue
        if on_disk:
            need = ([f for f, want in (("router_log.pt", router_log), ("delex_capture.pt", capture))
                     if want] or [])
            if any(not os.path.exists(r.path(f)) for f in need):
                continue
        out.append(r)
    return out


def get(name):
    return Run(name, _manifest_runs().get(name, set()))


def moe_layers(artifact):
    """MoE layer indices a loaded capture or router log actually contains, sorted.

    This is the only sanctioned source of a layer list. Callers iterate what the artifact holds and
    report anything they cannot cover; nobody writes `range(2, 10)`.
    """
    return sorted(int(x) for x in artifact["layers"].keys())


def selection(budget_order=("1e18", "1e19", "1e17", "1e16")):
    """The capture-sweep selection set (re-run plan Step 3 item 10).

    One run per cell, plus the dense control at each budget as the isoFLOP floor. Ordered by
    `budget_order`, which puts 1e18 first because no mechanistic measurement of any kind exists at
    that budget and it is where the temporal model wins.

    A cell is (budget, regime, granularity, ffn). `ffn` is in the key because the 1e18 panel has three
    shapes -- flame192 on the left flank, flame38m in the middle, flame512 on the right -- and
    granularity alone would collapse them into one cell and silently drop two thirds of the panel.

    Within a cell, prefer a run whose artifacts were already preserved (a capture or a router log
    marks the run the published analyses actually used), then the plainest recipe name, so a
    trigger-shaping or overlap variant never stands in for its cell.

    Excluded: runs with no preserved checkpoint (nothing to capture from), runs whose budget cannot be
    determined (parity and smoke-test runs, not science cells), and the dense controls -- a dense model
    has no experts and no router, so a routing capture records nothing. The dense floor belongs in the
    isoFLOP quality comparison instead.
    """
    best = {}
    for r in runs(ckpt=True):
        if r.budget == "unknown":
            continue
        if r.dense:
            continue        # no router: nothing for a routing capture to record
        key = (r.budget, r.regime, r.grain_label, r.meta.get("ffn"))
        # sort key: preserved artifacts first, then the shortest/plainest name
        rankr = (not (r.has_capture or r.has_router_log), len(r.name), r.name)
        cur = best.get(key)
        if cur is None or rankr < (not (cur.has_capture or cur.has_router_log),
                                   len(cur.name), cur.name):
            best[key] = r
    rank = {b: i for i, b in enumerate(budget_order)}
    return sorted(best.values(),
                  key=lambda r: (rank.get(r.budget, len(rank)), r.regime, r.grain_label, r.name))


if __name__ == "__main__":
    if "--selection" in sys.argv:
        sel = selection()
        if "--names-only" in sys.argv:
            print("\n".join(r.name for r in sel))
            sys.exit(0)
        print(f"capture-sweep selection set: {len(sel)} runs "
              f"(one per budget/regime/granularity cell, plus the dense floor)\n")
        print(f"{'run':34} {'budget':7} {'regime':9} {'grain':13} {'depth':5} capture?")
        for r in sel:
            print(f"{r.name:34} {r.budget:7} {r.regime:9} {r.grain_label:13} "
                  f"{str(depth_of(r.name) or '?'):5} {'already have' if r.has_capture else 'NEEDED'}")
        need = [r for r in sel if not r.has_capture]
        print(f"\n{len(need)} of {len(sel)} need a capture pass")
        sys.exit(0)

    rl = runs(router_log=True)
    cp = runs(capture=True)
    print(f"{len(_manifest_runs())} runs in MANIFEST.csv; "
          f"{len(rl)} with router_log.pt, {len(cp)} with delex_capture.pt, "
          f"{len(runs(ckpt=True))} with a checkpoint\n")
    print(f"{'run':34} {'regime':9} {'grain':13} {'budget':7} {'depth':5} artifacts")
    for r in sorted(rl + [c for c in cp if not c.has_router_log], key=lambda r: (r.budget, r.name)):
        art = ",".join(a for a, ok in (("log", r.has_router_log), ("cap", r.has_capture),
                                       ("ckpt", r.has_ckpt)) if ok)
        print(f"{r.name:34} {r.regime:9} {r.grain_label:13} {r.budget:7} "
              f"{str(r.depth or '?'):5} {art}")
    nb = [r for r in rl if not r.temporal]
    print(f"\nunconstrained runs with a router log (the only possible baseline replay arm): "
          f"{[r.name for r in nb] or 'NONE'}")
