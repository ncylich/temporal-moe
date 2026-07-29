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

__all__ = ["Run", "runs", "get", "moe_layers", "budget_of"]

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
        self.budget = self.meta.get("flops") or budget_of(name)
        self.depth = self.meta.get("L")         # transformer depth, or None for the 1e18 form

    # ---- artifact predicates: what analyses this run can actually feed ----
    @property
    def has_router_log(self):
        return "router_log.pt" in self.files

    @property
    def has_capture(self):
        return "delex_capture.pt" in self.files

    @property
    def has_ckpt(self):
        return "common.pt" in self.files

    def path(self, fname):
        return os.path.join(RUNS, self.name, fname)

    @property
    def regime(self):
        """'temporal' (rolling residency) or 'full' (unconstrained MoE)."""
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


# Transformer depth by shape, for runs whose own artifacts were never preserved and so have no
# run.meta on disk -- the 1e16/1e17 cells in mechinterp_locus.csv are all in this position. Each
# value is read from a run that IS in MANIFEST.csv and shares the shape, so nothing here is guessed:
#   s0     -> g3_tmoe_s0_1e16_ant0p02  L=4
#   s2     -> g3_tmoe_s2_1e17          L=6
#   s19opt -> moe_coarse_1e19          L=14
# Note s19opt is 14, not the 9 that both mechanism plan documents assume.
SHAPE_DEPTH = {"s0": 4, "s2": 6, "s19opt": 14}


def depth_of(name):
    """Transformer depth for a run, from its own run.meta if present, else from its shape."""
    r = get(name)
    if r.depth:
        return r.depth
    for shape, L in SHAPE_DEPTH.items():
        if re.search(rf"(^|_){shape}(_|$)", name):
            return L
    if "flame38m" in name:
        return 9              # the 38M fleet: 8 MoE layers 2-9 per e6_per_layer_ranking.csv
    return None


def budget_of(name):
    """Compute budget from the run name, for runs whose run.meta omits flops= (the 1e18 form)."""
    m = re.search(r"(1e1[6-9])", name)
    if m:
        return m.group(1)
    if "flame38m" in name:
        return "1e18"            # the 38M-active fleet is the 1e18 budget throughout
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


def runs(router_log=False, capture=False, ckpt=False, temporal=None, budget=None, on_disk=False):
    """Every run matching the filters, in name order. All filters default to off.

    router_log/capture/ckpt: require that artifact be preserved (per MANIFEST.csv).
    temporal: True for residency-constrained runs, False for unconstrained baselines.
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


if __name__ == "__main__":
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
