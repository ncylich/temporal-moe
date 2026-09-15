#!/usr/bin/env python3
"""A6-A9 -- structural routing statistics, per MoE layer.

Per (run, layer):
  PR_median          median expert selectivity: normalized inverse Simpson of renormalized gate mass
                     q_e(t)=g_e(t)/sum_t g_e(t);  PR_e = 1/(N sum_t q_e(t)^2) in (0,1]. 1 = the
                     expert draws uniformly from the stream, ~m/N = it lives on m positions.
  generalist_frac    |{e: PR_e>0.5}|/E
  router_entropy     mean_t[-sum_e g_e(t) ln g_e(t)] / ln E, per-token routing flatness in [0,1]
  eff_rank           participation ratio of the eigenvalues of the expert gate-mass covariance
  strong_corr_pairs  expert pairs whose per-token gate series correlate above 0.5
  dist2centroid_mean mean_e (1 - cos(w_e, wbar)) of flattened FFN weights against their centroid
  pairwise_cos_med   median / p99 of pairwise cos(w_e, w_e')

**The change is the `layer` key.** The published version pooled every expert of every MoE layer into
one row per model, discarding a layer key it already had, so selectivity, generalist fraction and
router entropy became depth curves for free (A6, A7, A9). Weight geometry (A8) is expected to be flat
with depth; it is computed per layer anyway so that "expected flat" is a measurement.

Gate statistics need only the capture. Weight geometry additionally needs the run's checkpoint, and
reading a Megatron distributed checkpoint needs `megatron` importable (see ckpt_read.py). When the
checkpoint is absent the gate columns are still written and the geometry columns are left blank with
the reason recorded, rather than failing the whole run.

    . scripts/env.sh
    PYTHONPATH="$ROOT/Megatron-LM:$ROOT" "$PY" analysis/probes/delex_structural.py

Bare positional arguments restrict the run to named cells. `--out=PATH` sends the table somewhere
other than the default; see `out_path` for why that is a flag rather than a constant.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry
import safe_csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

DEFAULT_OUT = os.path.join(ABLATIONS, "mechinterp_structural_1e19.csv")


def out_path():
    """Destination CSV; pass --out=PATH to write elsewhere.

    This was a module-level constant that someone repointed from `mechinterp_structural.csv` to the
    1e19 name. That silently orphaned the earlier file, which remains the only record of six runs —
    four of whose checkpoints are gone — and of pooled effective rank, a statistic the per-layer
    schema cannot express. Naming the destination on the command line makes overwriting one output by
    retargeting another impossible to do by accident.

    The default name says 1e19 but the file spans every budget from 1e16 up; the suffix is historical.
    """
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            return a.split("=", 1)[1]
    return DEFAULT_OUT
HEADER = ["label", "run", "budget", "regime", "grain", "layer", "E", "k",
          "PR_median", "generalist_frac", "router_entropy", "eff_rank", "strong_corr_pairs",
          "dist2centroid_mean", "pairwise_cos_med", "pairwise_cos_p99", "geometry_note"]


def gate_stats(d, L):
    """Per-layer gate statistics from one capture's layer record."""
    import torch
    Ld = d["layers"][L]
    lg = Ld["logits"].float()
    k = int(Ld["k"])
    E = lg.shape[-1]
    N = lg.shape[0] * lg.shape[1]
    g = torch.softmax(lg, dim=-1).reshape(N, E).double()
    H = float((-(g * g.clamp(min=1e-12).log()).sum(-1)).mean() / np.log(E))
    mass = g.sum(0)
    q = g / mass.clamp(min=1e-12)
    pr = (1.0 / (N * (q * q).sum(0)).clamp(min=1e-12)).numpy()
    gc = g - g.mean(0)
    cov = (gc.T @ gc) / N
    sd = cov.diag().clamp(min=1e-12).sqrt()
    cc = cov / (sd[:, None] * sd[None, :])
    iu = torch.triu_indices(E, E, 1)
    strong = int((cc[iu[0], iu[1]].abs() > 0.5).sum())
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    eff = float((ev.sum() ** 2) / (ev * ev).sum().clamp(min=1e-12))
    return dict(E=E, k=k, PR_median=float(np.median(pr)), generalist=float((pr > 0.5).mean()),
                entropy=H, eff_rank=eff, strong=strong)


def weight_geometry(run):
    """{layer: (dist2centroid_mean, pairwise_cos_med, pairwise_cos_p99)} from the checkpoint.

    Returns ({}, reason) when the checkpoint or the megatron import is unavailable.
    """
    import re
    import torch
    ck = os.path.join(registry.RUNS, run, "ckpt")
    if not os.path.isdir(ck):
        return {}, "no checkpoint on disk (scripts/artifacts.py pull --run %s)" % run
    try:
        import ckpt_read
        ip = ckpt_read.iter_dir(ck)
        meta = ckpt_read.weight_keys(ckpt_read.FileSystemReader(ip))
    except Exception as exc:                                   # megatron/TE import or DCP metadata
        return {}, f"checkpoint unreadable: {type(exc).__name__}: {exc}"[:160]
    keys = sorted(k for k in meta if re.search(r"experts\.experts\.linear_fc[12]\.weight$", k))
    if not keys:
        return {}, "checkpoint has no routed-expert weights"
    sd = ckpt_read.load(ip, keys)
    by_layer = {}
    for kk in keys:
        # +1: checkpoint module paths are 0-based, capture layer keys are TopKRouter.layer_number,
        # which is 1-based. Using the raw index attributed each layer's weights to the layer above and
        # left the deepest MoE layer with no geometry at all -- the same off-by-one delex_probe.py and
        # delex_lens.py had. Verified against the checkpoint: expert modules occupy 1..L-1 where the
        # capture holds 2..L.
        L = int(re.search(r"layers\.(\d+)\.", kk).group(1)) + 1
        t = sd[kk].float()
        by_layer.setdefault(L, []).append(t.reshape(t.shape[0], -1))
    out = {}
    for L, parts in by_layer.items():
        W = torch.cat(parts, dim=1)
        Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cbar = Wn.mean(0)
        cbar = cbar / cbar.norm().clamp(min=1e-12)
        C = Wn @ Wn.T
        iu = torch.triu_indices(W.shape[0], W.shape[0], 1)
        pc = C[iu[0], iu[1]]
        out[L] = (float((1 - (Wn @ cbar)).mean()), float(pc.median()),
                  float(np.percentile(pc.numpy(), 99)))
    return out, ""


def main():
    import torch
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    cells = [r for r in registry.runs(capture=True)
             if (not only or r.name in only) and os.path.exists(r.path("delex_capture.pt"))]
    if not cells:
        sys.exit("no captures on disk")
    rows = []
    for r in cells:
        d = torch.load(r.path("delex_capture.pt"), map_location="cpu", weights_only=False)
        layers = registry.moe_layers(d)
        if not layers:
            print(f"[warn] {r.name}: capture holds no MoE layers, skipping (rerun its capture)")
            continue
        geom, note = weight_geometry(r.name)
        if note:
            print(f"[warn] {r.name}: weight geometry (A8) unavailable — {note}")
        print(f"[run] {r.name} ({r.regime}, {r.grain_label}, {r.budget}) "
              f"layers {layers[0]}-{layers[-1]}", flush=True)
        for L in layers:
            s = gate_stats(d, L)
            g = geom.get(L)
            missing = "" if g else (note or f"layer {L} absent from checkpoint expert weights")
            rows.append([r.name, r.name, r.budget, r.regime, r.grain_label, L, s["E"], s["k"],
                         round(s["PR_median"], 4), round(s["generalist"], 4), round(s["entropy"], 4),
                         round(s["eff_rank"], 2), s["strong"],
                         round(g[0], 4) if g else "", round(g[1], 4) if g else "",
                         round(g[2], 4) if g else "", missing])
            print(f"    L{L:<3} PR_med={s['PR_median']:.3f} generalist={s['generalist']*100:4.0f}% "
                  f"Hbar={s['entropy']:.3f} eff_rank={s['eff_rank']:6.2f} "
                  f"strong_pairs={s['strong']:5d}"
                  + (f"  d2c={g[0]:.3f} cos_med={g[1]:.4f}" if g else "  [geometry: n/a]"),
                  flush=True)

    out = out_path()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    safe_csv.guard(out, rows, key_index=HEADER.index("run") if "run" in HEADER else None)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[write] {out}: {len(rows)} rows")
    nogeom = sum(1 for x in rows if x[-1])
    if nogeom:
        print(f"note: {nogeom}/{len(rows)} rows have no weight-geometry (A8) columns; the reason is "
              f"in the geometry_note column of each")


if __name__ == "__main__":
    main()
