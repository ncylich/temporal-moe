#!/usr/bin/env python3
"""OLMoE memory/bandwidth frontier: R x swaps grid on base + adapted surfaces, per-layer
damage curves, and a fitted min-cost per-layer allocation, all training-free in one load.

Stages (all rows appended to frontier_olmoe.csv as they land; a crash loses nothing prior):
  smoke    known-value anchors + machinery checks (R_map==scalar parity, s>1 raises swap rate,
           adapted surface loads and beats base). ANY smoke failure aborts before the long part.
  grid     surface {base, distill100M} x R {8,12,16,24,32} x s {1,2,4} -> bpb, swap_rate.
  layers   d_l(R): layer l constrained at R in {8,12,16,24}, all other layers free. Base surface.
  alloc    power-law fit per layer, greedy marginal allocation at slot budgets {192, 256}
           (1x = 128 slots has no freedom: R >= k = 8), validated jointly via R_map on base and
           on the adapted surface, against uniform R=12 / R=16 at iso-memory.

    frontier_olmoe.py            # full run
    frontier_olmoe.py --smoke    # smoke stage only
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
import residency as RES                                              # noqa: E402
from olmoe_paths import DATA_DIR                                     # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

CSURF = "/workspace/olmoe-adapt/data/csurf_distill100M_T1_lr3e-5_at100M.pt"
OUT = os.path.join(ABLATIONS, "frontier_olmoe.csv")
R_GRID, S_GRID, R_LAYER = [8, 12, 16, 24, 32], [1, 2, 4], [8, 12, 16, 24]
FIELDS = ["stage", "surface", "cell", "R", "swaps", "free_set", "R_map", "bpb",
          "swap_rate", "n_seq", "secs"]


def bpb(model, ids, divisor):
    tot = ntok = 0
    for i in range(ids.shape[0]):
        b = ids[i:i + 1].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        del lg
    return (tot / ntok) / divisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()

    D = json.load(open(f"{DATA_DIR}/bpb_slice_meta.json"))["divisor_D"]
    model, _ = RES.load_model()
    L = model.config.num_hidden_layers
    ids = torch.load(f"{DATA_DIR}/bpb_slice_ids.pt", weights_only=False)[: A.n_seq]
    ALL = list(range(L))

    # Adapted surface: same reconstruction as downstream.py (router + norms + expert LoRA r32
    # + attention LoRA r32, zipped against the checkpoint masters with a count assert).
    ck = torch.load(CSURF, map_location="cpu", weights_only=False)
    train_params = (RES.router_params(model) + RES.norm_params(model)
                    + RES.add_lora(model, r=32, alpha=64)
                    + RES.add_lora_attn(model, r=ck["lora_attn"], alpha=2 * ck["lora_attn"]))
    assert len(train_params) == len(ck["masters"]), \
        f"surface mismatch: {len(train_params)} params vs {len(ck['masters'])} masters"
    base_copy = [p.detach().clone() for p in train_params]

    def set_surface(name):
        src = ck["masters"] if name == "distill100M" else base_copy
        with torch.no_grad():
            for p, m in zip(train_params, src):
                p.data.copy_(m.to("cuda").to(p.dtype))

    model.eval()
    exists = os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if not exists:
        fh.write('"# OLMoE residency frontier, training-free, gate_mass=preserve, min_logit. '
                 'Surfaces: base and the 100M T=1 distillation winner (csurf_distill100M). '
                 'swap_rate = mean resident-set changes/token/layer (fetch-bandwidth axis); '
                 'memory axis = total resident slots (uniform R: R*16; R_map cells list slots). '
                 'Producer: analysis/ple/frontier_olmoe.py"\n')
        w.writeheader()

    def cell(stage, surface, name, R=8, swaps=1, free_set=None, r_map=None):
        RES._CFG.update(on=True, R=R, evict="min_logit", collect_telem=True,
                        gate_mass="preserve", swaps=swaps, R_map=r_map)
        RES.set_free_layers(free_set)
        RES.reset_telem()
        t0 = time.time()
        with torch.no_grad():
            v = bpb(model, ids, D)
        sw, _ = RES.telem_summary(model.config.num_experts)
        w.writerow({"stage": stage, "surface": surface, "cell": name, "R": R, "swaps": swaps,
                    "free_set": "" if free_set is None else ",".join(map(str, free_set)),
                    "R_map": "" if r_map is None else ";".join(f"{k}:{v2}" for k, v2 in
                                                               sorted(r_map.items())),
                    "bpb": f"{v:.6f}", "swap_rate": f"{sw:.4f}", "n_seq": A.n_seq,
                    "secs": f"{time.time()-t0:.1f}"})
        fh.flush()
        print(f"  [{stage}] {surface:12} {name:28} R={R} s={swaps} BPB={v:.6f} "
              f"swap={sw:.3f} ({time.time()-t0:.0f}s)", flush=True)
        return v

    # ---- smoke: every mechanism this run depends on, against known anchors ----
    set_surface("base")
    v_free = cell("smoke", "base", "free", free_set=ALL)
    assert abs(v_free - 0.6727) < 0.02, f"free anchor off: {v_free}"
    v_r8 = cell("smoke", "base", "R8_s1")
    assert abs(v_r8 - 0.8393) < 0.02, f"impose anchor off: {v_r8}"
    v_map = cell("smoke", "base", "R8_via_uniform_R_map", r_map={i: 8 for i in ALL})
    assert abs(v_map - v_r8) < 1e-5, f"R_map parity broken: {v_map} vs {v_r8}"
    v_s4 = cell("smoke", "base", "R8_s4", swaps=4)
    assert v_s4 < v_r8 - 0.005, f"s=4 did not help: {v_s4} vs {v_r8}"
    set_surface("distill100M")
    v_ad = cell("smoke", "distill100M", "R8_s1")
    assert abs(v_ad - 0.7779) < 0.02, f"adapted anchor off: {v_ad}"
    print("  [smoke] ALL SMOKES PASS", flush=True)
    if A.smoke:
        fh.close()
        return

    # ---- grid ----
    for surface in ("base", "distill100M"):
        set_surface(surface)
        for R in R_GRID:
            for s in S_GRID:
                cell("grid", surface, f"R{R}_s{s}", R=R, swaps=s)

    # ---- per-layer damage curves (base surface, others free) ----
    set_surface("base")
    d = {}
    for l in range(L):
        for R in R_LAYER:
            free = [x for x in ALL if x != l]
            d[(l, R)] = cell("layers", "base", f"solo_L{l:02d}_R{R}", R=R, free_set=free) - v_free

    # ---- fit + greedy allocation + joint validation ----
    fit = {}
    for l in range(L):
        xs = [math.log(R) for R in R_LAYER]
        ys = [math.log(max(d[(l, R)], 1e-6)) for R in R_LAYER]
        n = len(xs)
        b = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / \
            (n * sum(x * x for x in xs) - sum(xs) ** 2)
        a = (sum(ys) - b * sum(xs)) / n
        fit[l] = (math.exp(a), -b)                       # d_l(R) ~ A * R^-g
    pred = lambda l, R: fit[l][0] * R ** (-fit[l][1])
    for budget in (192, 256):
        alloc = {l: 8 for l in range(L)}
        for _ in range(budget - 8 * L):
            best = max(range(L), key=lambda l: pred(l, alloc[l]) - pred(l, alloc[l] + 1))
            alloc[best] += 1
        print(f"  [alloc] budget={budget}: " +
              " ".join(f"L{l}:{alloc[l]}" for l in range(L)), flush=True)
        for surface in ("base", "distill100M"):
            set_surface(surface)
            cell("alloc", surface, f"fitted_B{budget}", r_map=alloc)
            cell("alloc", surface, f"uniform_B{budget}", R=budget // L)
    fh.close()
    print("FRONTIER OLMOE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
