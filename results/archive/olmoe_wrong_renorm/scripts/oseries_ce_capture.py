#!/usr/bin/env python3
"""O-3 gate capture (orch 0130-2): free forward of the CE-ADAPTED model (merged_ce_model) over the same
24 audited-slice packs; store per-layer per-token top-24 plain-softmax mass (reward field, never renorm).
Feeds the CPU MinFlow solve for the CE-oracle-vs-scan gate. Writes data/oseries_ce/{idx.u8,val.f16,meta.json}.
Usage: oseries_ce_capture.py [n_packs]  (default 24)."""
import os, sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEQ = 4096; TOPK = 24
N_SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 24
OUT = "/workspace/olmoe-adapt/data/oseries_ce"; os.makedirs(OUT, exist_ok=True)
PATH = "/workspace/olmoe-adapt/merged_ce_model"                # CE-adapted (router+norms+LoRA baked)
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
L = model.config.num_hidden_layers; E = model.config.num_experts
print(f"[ce-cap] CE model loaded bf16 FREE-routing. L={L} E={E} N_SEQ={N_SEQ} top{TOPK}", flush=True)

ids = torch.load("/workspace/olmoe-adapt/data/bpb_slice_ids.pt")[:N_SEQ].long()
idx_mm = np.memmap(f"{OUT}/idx.u8", dtype=np.uint8, mode="w+", shape=(N_SEQ, L, SEQ, TOPK))
val_mm = np.memmap(f"{OUT}/val.f16", dtype=np.float16, mode="w+", shape=(N_SEQ, L, SEQ, TOPK))
rank_mass = torch.zeros(E, dtype=torch.float64); n_tok = 0
with torch.no_grad():
    for s in range(N_SEQ):
        x = ids[s:s + 1].to("cuda")
        out = model(x, output_router_logits=True)
        for l, rl in enumerate(out.router_logits):
            p = torch.softmax(rl.float(), dim=-1)              # CE model's own free softmax mass over 64
            sv, si = torch.sort(p, dim=-1, descending=True)
            rank_mass += sv.sum(0).double().cpu()
            idx_mm[s, l] = si[:, :TOPK].to(torch.uint8).cpu().numpy()
            val_mm[s, l] = sv[:, :TOPK].to(torch.float16).cpu().numpy()
        n_tok += x.shape[1]
        print(f"[ce-cap] {s+1}/{N_SEQ} packs", flush=True)
idx_mm.flush(); val_mm.flush()
rank_mass /= n_tok; cum = torch.cumsum(rank_mass, 0)
meta = {"n_seq": N_SEQ, "n_tokens": n_tok, "L": L, "E": E, "topk": TOPK, "seq": SEQ,
        "model": "merged_ce_model (CE-adapted)", "routing": "free_no_residency",
        "coverage_top8": float(cum[7]), "coverage_top24": float(cum[23])}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print(f"[ce-cap] coverage top8={float(cum[7]):.5f} top24={float(cum[23]):.5f}; DONE -> {OUT}", flush=True)
