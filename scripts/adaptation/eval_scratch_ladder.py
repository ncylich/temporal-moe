#!/usr/bin/env python3
"""Scratch-ladder (orch 0095): eval intermediate OLMoE-1B-7B-0924 pretraining checkpoints (FREE routing,
NO residency) on the Stage-1 audited slice (D=3.1089) to locate the from-scratch crossing vs our adapted
numbers (CE 0.8147=93.2%, C 0.8505=91.4%). Appends rows source=ckpt_ladder to olmoe_scratch_ladder.csv."""
import sys, json, torch, csv, os
from transformers import AutoModelForCausalLM
from datasets import config as _c

REPO = "allenai/OLMoE-1B-7B-0924"
BRANCHES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "step10000-tokens41B", "step25000-tokens104B", "step55000-tokens230B", "step125000-tokens524B"]
OUT = "/workspace/olmoe-adapt/data"
D = json.load(open(f"{OUT}/bpb_slice_meta.json"))["divisor_D"]
CSV = "/workspace/FLAME-MoE/results/ablations/olmoe_scratch_ladder.csv"
bpb_ids = torch.load(f"{OUT}/bpb_slice_ids.pt")
eval_sub = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, 256).long()].long()


def eval_bpb(model):
    tot = n = 0
    with torch.no_grad():
        for i in range(eval_sub.shape[0]):
            x = eval_sub[i:i + 1].to("cuda")
            out = model(x).logits.float()
            l = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)),
                                                  x[:, 1:].reshape(-1), reduction="sum")
            tot += l.item(); n += x[:, 1:].numel()
    return tot / n, (tot / n) / D


rows = []
for br in BRANCHES:
    tokB = br.split("tokens")[1].rstrip("B")
    print(f"[scratch] loading {REPO}@{br} ...", flush=True)
    try:
        m = AutoModelForCausalLM.from_pretrained(REPO, revision=br, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    except Exception as e:
        print(f"[scratch] FAILED {br}: {str(e)[:140]}", flush=True); continue
    ce, bpb = eval_bpb(m)
    rec = float(1 - (bpb - 0.6727) / (2.7507 - 0.6727))
    print(f"[scratch] {br}: tokens={tokB}B  CE={ce:.4f}  BPB={bpb:.4f}  (free routing)", flush=True)
    rows.append(("ckpt_ladder", f"0924@{br}", tokB, "audited_slice", f"{ce:.4f}", f"{bpb:.4f}",
                 f"free top8 no-residency bf16; from-scratch pretraining ckpt; vs adapted CE 0.8147/C 0.8505"))
    del m; torch.cuda.empty_cache()

with open(CSV, "a", newline="") as f:
    w = csv.writer(f)
    for r in rows:
        w.writerow(r)
print(f"[scratch] appended {len(rows)} ckpt_ladder rows to {CSV}", flush=True)
for r in rows:
    print("  ", r[1], "tokens", r[2] + "B", "BPB", r[5], flush=True)
