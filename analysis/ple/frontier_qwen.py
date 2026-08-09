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
R_GRID, S_GRID = [8, 12, 16, 24, 32], [1, 2, 4]
FIELDS = ["stage", "surface", "cell", "R", "swaps", "bpb", "swap_rate", "n_seq", "secs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("qwen3", "qwen3_5"))
    ap.add_argument("--eval-seq", type=int, default=16)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
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

    RU.install(model)
    RES.set_free_layers(None)
    model.eval()
    bpb_ids = torch.load(f"{FAM['data']}/bpb_slice_ids_{FAM['suffix']}.pt",
                         weights_only=False)[: A.eval_seq]

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
