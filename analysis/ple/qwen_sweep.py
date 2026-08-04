#!/usr/bin/env python3
"""Qwen3.5-35B-A3B-Base: correctness preflight + the whole test-time residency suite in one load.

A 35B checkpoint costs minutes to load and ~70 GB to hold, so the measurement design is one load and
many configurations replayed over the SAME cached batches, rather than one process per cell. Every
number below is therefore matched on input by construction, not by convention.

PREFLIGHT (hard asserts -- the suite refuses to report numbers if these fail):

    parity      residency off must be bitwise identical to the stock router. This is the claim that
                licenses every other number: if the hook perturbs the model when it is supposed to be
                inert, no downstream difference can be attributed to the constraint.
    R=E no-op   masking with R = num_experts must also be bitwise identical, since the resident set
                is then everything. Catches an off-by-one in the scan that parity alone would miss.
    resident R  the mask must leave exactly R finite logits per token, no more and no fewer.

MEASUREMENTS, all on the audited slice re-tokenized to byte-identical text (build_qwen_slice.py):

    anchors     free (unconstrained) and fully-constrained BPB. These bracket everything else; the
                per-layer numbers are only interpretable between them.
    per-layer   constrain exactly one MoE layer, leave the other 39 free. This is the advisor's
                test-time-only question and needs no training, so it is unaffected by this model
                shipping instruct-free base weights.
    matched-R   the same sweep at R=32. OLMoE ran R=k=8 of 64 = 12.5% resident; the same rule on 256
                experts is 3.1%, so "same recipe" is 4x harsher here. R=32 restores the FRACTION.
                Reporting only one of these would be an unmatched comparison.
    recipe      free {0,1,L-2,L-1} -- the first-two/last-two set that won on OLMoE.

Every cell also records effective experts (load and token senses) and the observed swap rate.

    qwen_sweep.py --n-seq 32
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402

DATA = "/workspace/qwen35-adapt/data"
OUT = "/workspace/qwen35-adapt/results"


def batches(n_seq, mb):
    ids = torch.load(os.path.join(DATA, "bpb_slice_ids_qwen.pt"), weights_only=False)[:n_seq]
    return [ids[i:i + mb].long() for i in range(0, len(ids), mb)]


@torch.no_grad()
def score(model, bl, divisor, free_set, R, want_eff=False):
    """Mean CE over the cached batches under one residency configuration.

    free_set None means 'constrain every layer'; a full list means 'constrain none'. Loss is
    computed here rather than by passing labels, because passing labels would make the model add its
    own auxiliary term and silently change what BPB means.
    """
    RES._CFG.update(on=True, R=R, evict="min_logit", collect_telem=True)
    RES.set_free_layers(free_set)
    RES.reset_telem()
    tot, ntok = 0.0, 0
    eff_acc = None
    for b in bl:
        b = b.to("cuda")
        if want_eff:
            RQ.capture(True)
        out = model(b)
        lg = out.logits[:, :-1]
        tg = b[:, 1:]
        # Chunked: Qwen's vocab is 248320, so one float copy of a 4096-token logit tensor is 4.07 GB
        # on top of 70 GB of weights. Casting a slice at a time keeps the peak near 0.5 GB and does
        # not change the result -- reduction='sum' is exactly additive over a partition of the tokens.
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        if want_eff:
            e = RES.effective_experts(RQ.captured(), b.shape[0], b.shape[1], R)
            eff_acc = e if eff_acc is None else [
                {**a, "eff_load": a["eff_load"] + c["eff_load"], "eff_tok": a["eff_tok"] + c["eff_tok"]}
                for a, c in zip(eff_acc, e)]
            RQ.capture(False)
        del out, lg
    n = len(bl)
    if eff_acc is not None:
        eff_acc = [{**a, "eff_load": a["eff_load"] / n, "eff_tok": a["eff_tok"] / n} for a in eff_acc]
    ce_nats = tot / ntok
    swap, ent = RES.telem_summary(model.config.num_experts)
    return ce_nats / divisor, ce_nats, swap, ent, eff_acc


@torch.no_grad()
def preflight(model, bl, R_all):
    """Three asserts that must hold before any number is worth recording.

    no_grad is not an optimisation here. Without it these three forwards retain an autograd graph
    across 40 layers and 256 experts for a comparison that takes no gradients, which alone was enough
    to exhaust an 80 GB card on top of a 69 GB model.
    """
    # 512 tokens, not 4096: this holds a reference logits tensor for an exact comparison, and at
    # full length that copy alone is 2 GB in bf16. Parity is a property of the hook, not of length.
    b = bl[0][:1, :512].to("cuda")
    RES._CFG.update(on=False, collect_telem=False)
    ref = model(b).logits.float().clone()

    RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False)
    RES.set_free_layers(list(range(R_all["n_layers"])))            # every layer free == inert
    d_free = float((model(b).logits.float() - ref).abs().max())

    RES.set_free_layers(None)
    RES._CFG.update(R=R_all["E"])                                  # resident set == all experts
    d_full = float((model(b).logits.float() - ref).abs().max())

    RES._CFG.update(R=8)
    RQ.capture(True); model(b); lg = RQ.captured(); RQ.capture(False)
    N, E = lg[0].shape
    mask = RES.compute_resident_mask_accel(
        lg[0].float().view(1, N, E).transpose(0, 1).contiguous(), 8, evict="min_logit")
    per_tok = mask.sum(-1).float()
    print(f"  [preflight] residency-off vs stock  max|dlogit| = {d_free:.3e}")
    print(f"  [preflight] R=E vs stock            max|dlogit| = {d_full:.3e}")
    print(f"  [preflight] resident per token      min {per_tok.min():.0f} max {per_tok.max():.0f} (want 8)")
    assert d_free == 0.0, f"hook is not inert when disabled (max|dlogit|={d_free:.3e})"
    assert d_full == 0.0, f"R=E is not a no-op (max|dlogit|={d_full:.3e})"
    assert per_tok.min() == 8 and per_tok.max() == 8, "resident set is not exactly R per token"
    return d_free, d_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--skip-matched", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0,
                    help="limit the per-layer sweep (smoke runs); 0 = every layer")
    ap.add_argument("--tag", default="")
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    meta = json.load(open(os.path.join(DATA, "bpb_slice_meta_qwen.json")))
    D = meta["divisor_D"]
    model, tok = RQ.load_model()
    cfg = model.config
    E, k, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
    bl = batches(A.n_seq, A.mb)
    print(f"  slice: {A.n_seq} seq x {meta['seq']} tok, divisor {D:.7f}", flush=True)

    preflight(model, bl, {"E": E, "n_layers": L})

    rows = []
    def cell(name, free_set, R, want_eff=False):
        t0 = time.time()
        bpb, ce, swap, ent, eff = score(model, bl, D, free_set, R, want_eff)
        frac = RQ.resident_fraction(cfg, R)
        rows.append({"cell": name, "R": R, "resident_routed_pct": f"{100*frac['routed']:.2f}",
                     "resident_with_shared_pct": f"{100*frac['with_shared']:.2f}",
                     "bpb": f"{bpb:.6f}", "ce_nats": f"{ce:.6f}", "swap_rate": f"{swap:.4f}",
                     "route_entropy": f"{ent:.4f}",
                     "eff_load_med": (f"{sorted(x['eff_load'] for x in eff)[len(eff)//2]:.2f}"
                                      if eff else ""),
                     "n_seq": A.n_seq, "secs": f"{time.time()-t0:.1f}"})
        print(f"  {name:26} R={R:<3} BPB={bpb:.6f} swap={swap:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if eff:
            with open(os.path.join(OUT, f"eff_{name}.json"), "w") as f:
                json.dump(eff, f, indent=1)
        return bpb

    ALL = list(range(L))
    free_bpb = cell("free_baseline", ALL, 8, want_eff=True)
    cell("constrained_all_R8", None, 8, want_eff=True)
    if not A.skip_matched:
        cell("constrained_all_R32", None, 32, want_eff=True)
    cell("recipe_free_0_1_last2", [0, 1, L - 2, L - 1], 8)

    sweep = list(range(L)) if not A.max_layers else list(range(min(A.max_layers, L)))
    for li in sweep:                                               # constrain exactly one layer
        cell(f"solo_L{li:02d}_R8", [x for x in ALL if x != li], 8)
    if not A.skip_matched:
        for li in sweep:
            cell(f"solo_L{li:02d}_R32", [x for x in ALL if x != li], 32)

    path = os.path.join(OUT, f"qwen35_residency_suite{A.tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# Qwen3.5-35B-A3B-Base test-time residency suite, single model load, all cells "
                    f"replayed over identical cached batches. E={E} k={k} layers={L}. BPB divisor "
                    f"{D:.7f} (ln2 x bytes/token) on the audited slice re-tokenized to byte-identical "
                    f"text. solo_Lxx = that layer constrained, all others free. 'damage' is that "
                    f"cell's BPB minus free_baseline ({free_bpb:.6f}). resident_with_shared includes "
                    f"Qwen's always-on shared expert. Producer: analysis/ple/qwen_sweep.py"])
        w.writerow(list(rows[0].keys()) + ["damage_vs_free"])
        for r in rows:
            w.writerow(list(r.values()) + [f"{float(r['bpb'])-free_bpb:+.6f}"])
    print(f"\n[write] {path}: {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
