#!/usr/bin/env python3
"""Qwen memory/bandwidth frontier on the unsloth path: R x swaps grid, base + adapted winner.

Same design as frontier_olmoe.py (smoke anchors abort before the long part; every row is
flushed as it lands). 'base' is the zero-init LoRA surface (exact no-op), 'distill100M' copies
the campaign adapter tensors in; both scored by TQ.evaluate on the audited slice.

    frontier_qwen.py --family qwen3_5 [--smoke]
"""
import argparse
import csv
import json
import os
import sys
import time

import unsloth  # noqa: F401  must precede transformers
from unsloth import FastModel
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                              # noqa: E402
import residency_unsloth as RU                                       # noqa: E402
import train_qwen as TQ                                              # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

ADAPTER = {"qwen3": ("unsloth_distill100M_T1_lr1e-4_adapter.pt", 0.667648, 0.7337),
           "qwen3_5": ("unsloth_distill100M_T1_lr3e-5_adapter.pt", 0.662826, 0.6799)}
R_GRID, S_GRID, R_LAYER = [8, 12, 16, 24, 32], [1, 2, 4], [8, 12, 16, 24]
FIELDS = ["stage", "surface", "cell", "R", "swaps", "bpb", "swap_rate", "n_seq", "secs"]
PL_FIELDS = ["stage", "surface", "cell", "R", "free_set", "R_map", "bpb", "swap_rate",
             "n_seq", "secs"]


def perlayer(A, model, set_surface, bpb_ids, D, L, expect_impose):
    """d_l(R) curves (solo layer constrained, others free), power-law fit, greedy allocation
    validated jointly via R_map against uniform at iso-memory. Mirror of frontier_olmoe.py's
    layers/alloc stages; all layers measured (cells are ~6s on this path, no subsampling)."""
    import math
    ALL = list(range(L))
    out = os.path.join(ABLATIONS, f"perlayer_{A.family}.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=PL_FIELDS)
    if not exists:
        fh.write(f'"# {A.family} per-layer damage d_l(R): solo layer l constrained at R, others '
                 f'free, base surface; then fitted greedy allocation vs uniform at iso-memory '
                 f'slot budgets, both surfaces. Training-free, min_logit <=1 swap/token, '
                 f'gate_mass=preserve. free_set column: ALL = every layer free, all_but_l = solo '
                 f'cell. Producer: analysis/ple/frontier_qwen.py --perlayer"\n')
        w.writeheader()

    def cell(stage, surface, name, R=8, free_set=None, fs_tag="", r_map=None):
        RES._CFG.update(on=True, R=R, evict="min_logit", collect_telem=True, swaps=1,
                        R_map=r_map)
        RES.set_free_layers(free_set)
        t0 = time.time()
        v, sw, _ = TQ.evaluate(model, bpb_ids, D, A.mb)
        model.eval()
        w.writerow({"stage": stage, "surface": surface, "cell": name, "R": R,
                    "free_set": fs_tag,
                    "R_map": "" if r_map is None else ";".join(
                        f"{k}:{v2}" for k, v2 in sorted(r_map.items())),
                    "bpb": f"{v:.6f}", "swap_rate": f"{sw:.4f}", "n_seq": A.eval_seq,
                    "secs": f"{time.time()-t0:.1f}"})
        fh.flush()
        print(f"  [{stage}] {surface:12} {name:24} R={R} BPB={v:.6f} swap={sw:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return v

    # smoke: free anchor, impose anchor, R_map parity through the NEW plumbing, solo sanity
    set_surface("base")
    v_free = cell("smoke", "base", "free", free_set=ALL, fs_tag="ALL")
    v_r8 = cell("smoke", "base", "R8_s1")
    assert abs(v_r8 - expect_impose) < 0.02, f"impose anchor off: {v_r8} vs {expect_impose}"
    assert v_free < v_r8 - 0.02, f"free anchor not below impose: {v_free} vs {v_r8}"
    v_map = cell("smoke", "base", "R8_via_uniform_R_map", r_map={i: 8 for i in ALL})
    # NOT 1e-5 as on the OLMoE path: the unsloth experts forward accumulates with atomic
    # index_add_, which is run-to-run nondeterministic at ~2-3e-4 BPB (the committed frontier
    # R8 cell moves that much across identical runs). 1e-3 still catches any real R_map error,
    # the smallest of which (off-by-one layer index) measures ~2e-2.
    assert abs(v_map - v_r8) < 1e-3, f"R_map parity broken: {v_map} vs {v_r8}"
    v_solo = cell("smoke", "base", "solo_L00_R8", free_set=[x for x in ALL if x != 0],
                  fs_tag="all_but_0")
    assert v_free - 0.01 < v_solo < v_r8, f"solo cell implausible: {v_solo}"
    print("  [smoke] ALL SMOKES PASS", flush=True)
    if A.smoke:
        fh.close()
        return

    # per-layer damage curves, base surface
    d = {(0, 8): v_solo - v_free}
    for l in range(L):
        for R in R_LAYER:
            if (l, R) in d:
                continue
            v = cell("layers", "base", f"solo_L{l:02d}_R{R}", R=R,
                     free_set=[x for x in ALL if x != l], fs_tag=f"all_but_{l}")
            d[(l, R)] = v - v_free

    # fit d_l(R) ~ A * R^-g per layer, greedy marginal allocation, joint validation
    fit = {}
    for l in range(L):
        xs = [math.log(R) for R in R_LAYER]
        ys = [math.log(max(d[(l, R)], 1e-6)) for R in R_LAYER]
        n = len(xs)
        b = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / \
            (n * sum(x * x for x in xs) - sum(xs) ** 2)
        a = (sum(ys) - b * sum(xs)) / n
        fit[l] = (math.exp(a), -b)
    pred = lambda l, R: fit[l][0] * R ** (-fit[l][1])                     # noqa: E731
    for budget in (12 * L, 16 * L):
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
    print(f"PERLAYER {A.family.upper()} COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("qwen3", "qwen3_5"))
    ap.add_argument("--eval-seq", type=int, default=16)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--r-list", default=None,
                    help="comma R list: run ONLY these grid cells (s=1, both surfaces)")
    ap.add_argument("--perlayer", action="store_true",
                    help="per-layer d_l(R) curves + fitted allocation (perlayer_{family}.csv)")
    A = ap.parse_args()
    adapter_file, expect_adapted, expect_impose = ADAPTER[A.family]

    FAM = TQ.resolve(A.family)
    D = json.load(open(f"{FAM['data']}/bpb_slice_meta_{FAM['suffix']}.json"))["divisor_D"]
    ck = torch.load(os.path.join(FAM["out"], adapter_file), map_location="cpu",
                    weights_only=False)
    model, _ = FastModel.from_pretrained(FAM["model"], max_seq_length=2048,
                                         dtype=torch.bfloat16, load_in_4bit=False,
                                         full_finetuning=False)
    for mod in model.modules():
        if getattr(mod, "visual", None) is not None and "Vision" in type(mod.visual).__name__:
            mod.visual = None
            torch.cuda.empty_cache()
            break
    model = FastModel.get_peft_model(model, r=32, lora_alpha=64, lora_dropout=0.0,
                                     use_gradient_checkpointing=False)
    params = dict(model.named_parameters())
    missing = [n for n in ck["tensors"] if n not in params]
    assert not missing, f"{len(missing)} adapter tensors unmatched, e.g. {missing[:3]}"
    base_copy = {n: params[n].detach().cpu().clone() for n in ck["tensors"]}

    def set_surface(name):
        src = ck["tensors"] if name == "distill100M" else base_copy
        with torch.no_grad():
            for n, t in src.items():
                params[n].data.copy_(t.to(params[n].dtype))

    L = RU.install(model)
    RES.set_free_layers(None)
    model.eval()
    bpb_ids = torch.load(f"{FAM['data']}/bpb_slice_ids_{FAM['suffix']}.pt",
                         weights_only=False)[: A.eval_seq]

    if A.perlayer:
        perlayer(A, model, set_surface, bpb_ids, D, L, expect_impose)
        return

    out = os.path.join(ABLATIONS, f"frontier_{A.family}.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if not exists:
        fh.write(f'"# {A.family} residency frontier, training-free, unsloth path, min_logit. '
                 f'Surfaces: base (zero-init LoRA no-op) and the 100M T=1 distillation adapter. '
                 f'swap_rate = resident changes/token/layer. Memory axis = R x n_layers slots. '
                 f'Producer: analysis/ple/frontier_qwen.py"\n')
        w.writeheader()

    def cell(stage, surface, name, R=8, swaps=1):
        RES._CFG.update(on=True, R=R, evict="min_logit", collect_telem=True, swaps=swaps,
                        R_map=None)
        t0 = time.time()
        v, sw, _ = TQ.evaluate(model, bpb_ids, D, A.mb)
        model.eval()
        w.writerow({"stage": stage, "surface": surface, "cell": name, "R": R, "swaps": swaps,
                    "bpb": f"{v:.6f}", "swap_rate": f"{sw:.4f}", "n_seq": A.eval_seq,
                    "secs": f"{time.time()-t0:.1f}"})
        fh.flush()
        print(f"  [{stage}] {surface:12} {name:12} R={R} s={swaps} BPB={v:.6f} "
              f"swap={sw:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return v

    if A.r_list:
        for surface in ("base", "distill100M"):
            set_surface(surface)
            for R in (int(x) for x in A.r_list.split(",")):
                cell("grid", surface, f"R{R}_s1", R=R, swaps=1)
        fh.close()
        print(f"FRONTIER {A.family.upper()} EXTRA CELLS COMPLETE", flush=True)
        return
    set_surface("base")
    v_r8 = cell("smoke", "base", "R8_s1")
    assert abs(v_r8 - expect_impose) < 0.02, f"impose anchor off: {v_r8} vs {expect_impose}"
    v_s4 = cell("smoke", "base", "R8_s4", swaps=4)
    assert v_s4 < v_r8 - 0.002, f"s=4 did not help: {v_s4} vs {v_r8}"
    set_surface("distill100M")
    v_ad = cell("smoke", "distill100M", "R8_s1")
    assert abs(v_ad - expect_adapted) < 0.005, f"adapter reload off: {v_ad} vs {expect_adapted}"
    print("  [smoke] ALL SMOKES PASS", flush=True)
    if A.smoke:
        fh.close()
        return

    for surface in ("base", "distill100M"):
        set_surface(surface)
        for R in R_GRID:
            for s in S_GRID:
                if (surface, R, s) in {("base", 8, 1), ("base", 8, 4),
                                       ("distill100M", 8, 1)}:
                    continue                                   # already measured by smoke
                cell("grid", surface, f"R{R}_s{s}", R=R, swaps=s)
    fh.close()
    print(f"FRONTIER {A.family.upper()} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
