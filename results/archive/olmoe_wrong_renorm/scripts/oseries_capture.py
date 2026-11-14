#!/usr/bin/env python3
"""O-0 capture (orch 0090/0097): free forward of base OLMoE-1B-7B-0125 (NO residency) over ~N tokens;
store per-layer per-token top-24 softmax-mass fields (the reward field, NEVER renormalized) + a global
rank-mass histogram to verify top-24 covers >=99.5% cumulative mass. Fields feed the CPU MinFlow ladder
(replay with zero model evals). Reward = the frozen base's plain softmax mass over all 64 experts.

Usage: oseries_capture.py <n_tokens>   (default 2M). Writes data/oseries/{idx.u8,val.f16,meta.json}.
"""
import os, sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

SEQ = 4096; TOPK = 24
N_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
N_SEQ = N_TOK // SEQ
OUT = "/workspace/olmoe-adapt/data/oseries"; os.makedirs(OUT, exist_ok=True)
PATH = "/workspace/olmoe-adapt/model"
tok = AutoTokenizer.from_pretrained(PATH); EOS = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
L = model.config.num_hidden_layers; E = model.config.num_experts
print(f"[o0] model loaded bf16 free-routing. L={L} E={E} capture N_SEQ={N_SEQ} (~{N_TOK} tok), top{TOPK}", flush=True)

# corpus = the Stage-1 audited slice ids (in-distribution held-out) — reuse for consistency with BPB work
ids = torch.load("/workspace/olmoe-adapt/data/bpb_slice_ids.pt")[:N_SEQ].long()
print(f"[o0] using {ids.shape[0]} audited-slice packs of {SEQ}", flush=True)

idx_mm = np.memmap(f"{OUT}/idx.u8", dtype=np.uint8, mode="w+", shape=(ids.shape[0], L, SEQ, TOPK))
val_mm = np.memmap(f"{OUT}/val.f16", dtype=np.float16, mode="w+", shape=(ids.shape[0], L, SEQ, TOPK))
rank_mass = torch.zeros(E, dtype=torch.float64)          # mean sorted-descending softmax mass per rank
n_tok_seen = 0
with torch.no_grad():
    for s in range(ids.shape[0]):
        x = ids[s:s + 1].to("cuda")
        out = model(x, output_router_logits=True)
        for l, rl in enumerate(out.router_logits):        # rl: [B*S, E]
            p = torch.softmax(rl.float(), dim=-1)          # plain softmax mass over all 64 (never renorm)
            sv, si = torch.sort(p, dim=-1, descending=True)
            rank_mass += sv.sum(0).double().cpu()          # accumulate for the histogram/coverage
            idx_mm[s, l] = si[:, :TOPK].to(torch.uint8).cpu().numpy()
            val_mm[s, l] = sv[:, :TOPK].to(torch.float16).cpu().numpy()
        n_tok_seen += x.shape[1]
        if (s + 1) % 25 == 0:
            print(f"[o0] {s+1}/{ids.shape[0]} seqs captured", flush=True)
idx_mm.flush(); val_mm.flush()

rank_mass /= n_tok_seen                                    # mean mass at each sorted rank
cum = torch.cumsum(rank_mass, 0)
cov_top8 = float(cum[7]); cov_top24 = float(cum[23])
meta = {"n_seq": ids.shape[0], "n_tokens": n_tok_seen, "L": L, "E": E, "topk": TOPK, "seq": SEQ,
        "corpus": "stage1 audited slice (dolmino dclm/024*)", "routing": "free_top8_no_residency",
        "coverage_top8": cov_top8, "coverage_top24": cov_top24,
        "rank_mass_top32": [round(float(v), 6) for v in rank_mass[:32]],
        "note": "reward = plain base softmax mass over 64 experts, never renormalized after top-24 truncation"}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print(f"[o0] rank-mass coverage: top8={cov_top8:.5f}  top24={cov_top24:.5f}  (need top24>=0.995)", flush=True)
print(f"[o0] DONE captured {ids.shape[0]} seqs -> {OUT}/idx.u8,val.f16", flush=True)
