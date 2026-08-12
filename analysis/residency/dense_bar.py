#!/usr/bin/env python3
"""The dense floor: BPB + ten-task downstream for a small dense model, on an MoE family's slice.

A temporally-constrained MoE earns its residency budget only if it beats the dense model of
comparable ACTIVE size you could run instead -- Qwen3-30B-A3B activates ~3B params, so
Qwen3-4B-Base is its honest alternative; likewise Qwen3.5-4B-Base for Qwen3.5. (OLMoE's
analogous bar, OLMo-1B, is already known to sit at the constrained model's level.)

BPB is computed on the SAME token ids and divisor as the family's audited slice -- valid
because each dense sibling shares its family's tokenizer -- so the number is directly
comparable to every figure in sweep_RESULTS.md. Downstream is the same ten-task 0-shot
suite, acc-only basis, limit 500.

    dense_bar.py --model /workspace/dense-bars/qwen3-4b --family qwen3 --tag qwen3_4b
"""
import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_qwen as TQ                                            # noqa: E402

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "sciq", "winogrande",
         "openbookqa", "boolq", "lambada_openai", "copa"]


@torch.no_grad()
def bpb_on_slice(model, ids, divisor, chunk=512):
    tot = ntok = 0
    for i in range(len(ids)):
        b = ids[i:i + 1].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], chunk):
            sl = lg[:, i0:i0 + chunk].float()
            tot += float(F.cross_entropy(sl.reshape(-1, sl.shape[-1]),
                                         tg[:, i0:i0 + chunk].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        del lg
    return (tot / ntok) / divisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=("qwen3", "qwen3_5"),
                    help="whose slice/divisor to score against (tokenizer-shared)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-seq", type=int, default=16)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    A = ap.parse_args()

    FAM = TQ.resolve(A.family)
    D = json.load(open(f"{FAM['data']}/bpb_slice_meta_{FAM['suffix']}.json"))["divisor_D"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model)
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16).to("cuda").eval()

    ids = torch.load(f"{FAM['data']}/bpb_slice_ids_{FAM['suffix']}.pt",
                     weights_only=False)[: A.eval_seq]
    bpb = bpb_on_slice(model, ids, D)
    print(f"  [dense-bar] {A.tag}: BPB {bpb:.6f} on the {A.family} audited slice", flush=True)
    torch.cuda.empty_cache()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    t0 = time.time()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=A.batch_size)
    results = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0,
                              limit=(A.limit or None), bootstrap_iters=1000)["results"]
    print(f"  [ds] scored {len(TASKS)} tasks in {(time.time()-t0)/60:.1f} min", flush=True)

    out = os.path.join(FAM["out"], f"ds_dense_bar_{A.tag}.csv")
    accs = {}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        f.write(f'"# Dense floor {A.tag} ({A.model}) scored on the {A.family} audited slice: '
                f'BPB {bpb:.6f}. Ten 0-shot tasks, acc, limit {A.limit}. '
                f'Producer: analysis/residency/dense_bar.py"\n')
        w.writerow(["task", "metric", A.tag, f"{A.tag}_stderr"])
        for t in TASKS:
            r = results.get(t, {})
            for m, se in (("acc,none", "acc_stderr,none"), ("acc_norm,none", "acc_norm_stderr,none")):
                if m in r:
                    sev = r.get(se)
                    accs[t] = r[m]
                    w.writerow([t, m.split(",")[0], f"{r[m]:.4f}",
                                f"{sev:.4f}" if isinstance(sev, float) else ""])
                    break
    print(f"  [ds] wrote {out}")
    print(f"  [dense-bar] {A.tag}: avg acc {sum(accs.values())/len(accs):.4f} over {len(accs)} tasks")
    print("=== DENSE BAR COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
