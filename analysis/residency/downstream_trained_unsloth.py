#!/usr/bin/env python3
"""Downstream suite for a train_unsloth.py winner: load the adapter on the unsloth path,
verify the reload reproduces the training-time BPB, then run the ten-task 0-shot suite.

Same shape as downstream.py (OLMoE) and qwen_downstream.py (untrained arms): the model is
configured once, then handed to lm_eval's HFLM — the residency hook is global state, so the
constraint in force during scoring is the one asserted here (R=8, all layers, min_logit).
The reload check is the same idea as downstream.py's --expect-bpb: if the recomputed BPB is
not within tolerance of the training-time value, the wrong weights are loaded and the run
aborts rather than producing plausible-but-wrong accuracies.

    downstream_trained_unsloth.py --family qwen3 --adapter unsloth_sweep_lr1e-4_adapter.pt \
        --expect-bpb 0.677108 --tag qwen3_lr1e-4_win
"""
import argparse
import csv
import json
import os
import sys
import time

import unsloth  # noqa: F401  must precede any transformers import
from unsloth import FastModel
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_unsloth as RU                                     # noqa: E402
import train_qwen as TQ                                            # noqa: E402

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "sciq", "winogrande",
         "openbookqa", "boolq", "lambada_openai", "copa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("qwen3", "qwen3_5"))
    ap.add_argument("--adapter", default=None, help="unsloth_*_adapter.pt in the family out dir")
    ap.add_argument("--base", action="store_true", help="score the zero-init (base) surface")
    ap.add_argument("--r-map", default=None, help="per-layer residency 'l:R;...'")
    ap.add_argument("--expect-bpb", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="reload tolerance; wider than OLMoE's 2e-4 because the training-time "
                         "value comes from a mid-loop eval a few steps before the saved state")
    ap.add_argument("--lora", type=int, default=32)
    ap.add_argument("--eval-seq", type=int, default=16)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--free-set", default="",
                    help="'' = constrained R8 everywhere (trained winners); 'all' = residency "
                         "inert (null arms must be scored in their training configuration)")
    A = ap.parse_args()

    FAM = TQ.resolve(A.family)
    D = json.load(open(f"{FAM['data']}/bpb_slice_meta_{FAM['suffix']}.json"))["divisor_D"]
    assert A.base or A.adapter, "need --adapter or --base"
    if A.base:
        ck, sd, ck_path = {"R": 8}, {}, "(base surface)"
    else:
        ck_path = A.adapter if os.path.isabs(A.adapter) else os.path.join(FAM["out"], A.adapter)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        sd = ck["tensors"]

    model, tok = FastModel.from_pretrained(
        FAM["model"], max_seq_length=2048, dtype=torch.bfloat16,
        load_in_4bit=False, full_finetuning=False)
    # Qwen3.5's composite checkpoint makes unsloth return a Processor; HFLM asserts on
    # tokenizer type, so unwrap to the inner tokenizer (no-op for plain tokenizers).
    tok = getattr(tok, "tokenizer", tok)
    for mod in model.modules():
        if getattr(mod, "visual", None) is not None and "Vision" in type(mod.visual).__name__:
            mod.visual = None
            torch.cuda.empty_cache()
            break
    model = FastModel.get_peft_model(
        model, r=A.lora, lora_alpha=2 * A.lora, lora_dropout=0.0,
        use_gradient_checkpointing=False)

    # Reload the trained surface. Names must match the save-time named_parameters exactly;
    # a miss means the construction diverged from training and the numbers would be wrong.
    params = dict(model.named_parameters())
    missing = [n for n in sd if n not in params]
    if missing:
        sys.exit(f"FATAL: {len(missing)} adapter tensors have no destination, e.g. {missing[:3]}")
    with torch.no_grad():
        for n, t in sd.items():
            params[n].data.copy_(t.to(params[n].dtype))
    print(("  [reload] BASE surface (zero-init LoRA no-op)" if A.base else
           f"  [reload] {len(sd)} tensors -> model "
           f"({sum(t.numel() for t in sd.values())/1e6:.1f}M)"), flush=True)

    RU.install(model)
    rmap = ({int(p.split(":")[0]): int(p.split(":")[1]) for p in A.r_map.split(";")}
            if A.r_map else None)
    RES._CFG.update(on=True, R=ck.get("R", 8), evict="min_logit", collect_telem=True,
                    R_map=rmap)
    if A.free_set == "all":
        L = getattr(model.config, "text_config", model.config).num_hidden_layers
        RES.set_free_layers(list(range(L)))
    else:
        RES.set_free_layers(None)
    model.eval()

    bpb_ids = torch.load(f"{FAM['data']}/bpb_slice_ids_{FAM['suffix']}.pt",
                         weights_only=False)[: A.eval_seq]
    # mb=2: measured batch-invariant on this stack (bs1-vs-bs2 agreement exactly 1.0),
    # halves the reload-check wall time.
    bpb, swap, ent = TQ.evaluate(model, bpb_ids, D, 2)
    model.eval()
    dev = bpb - A.expect_bpb
    print(f"  [reload-check] BPB {bpb:.6f} vs expected {A.expect_bpb:.6f} (dev {dev:+.2e}, "
          f"tol {A.tol:.0e})  swap={swap:.4f}", flush=True)
    if abs(dev) > A.tol:
        sys.exit("FATAL: reload check failed -- wrong weights; aborting before scoring")
    torch.cuda.empty_cache()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    t0 = time.time()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=A.batch_size)
    results = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0,
                              limit=(A.limit or None), bootstrap_iters=1000)["results"]
    print(f"  [ds] scored {len(TASKS)} tasks in {(time.time()-t0)/60:.1f} min", flush=True)

    out = os.path.join(FAM["out"], f"ds_trained_{A.tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        f.write(f'"# {A.family} trained winner downstream, unsloth path, R=8 all layers, '
                f'adapter {os.path.basename(ck_path) if os.sep in ck_path else ck_path}, reload BPB {bpb:.6f} '
                f'(expected {A.expect_bpb:.6f}). Ten 0-shot tasks, limit {A.limit}. '
                f'Producer: analysis/residency/downstream_trained_unsloth.py"\n')
        w.writerow(["task", "metric", A.tag, f"{A.tag}_stderr"])
        for t in TASKS:
            r = results.get(t, {})
            for m, se in (("acc,none", "acc_stderr,none"), ("acc_norm,none", "acc_norm_stderr,none")):
                if m in r:
                    sev = r.get(se)
                    w.writerow([t, m.split(",")[0], f"{r[m]:.4f}",
                                f"{sev:.4f}" if isinstance(sev, float) else ""])
                    break
    print(f"  [ds] wrote {out}", flush=True)
    print(json.dumps({t: results[t].get("acc,none", results[t].get("acc_norm,none"))
                      for t in TASKS if t in results}, indent=2))
    print("=== TRAINED DOWNSTREAM COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
