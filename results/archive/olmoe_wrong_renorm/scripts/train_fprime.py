#!/usr/bin/env python3
"""Arm F-prime (per orch 0074/0078/0079): the capacity-floor probe. Warm-start from the BEST recipe's
trained deltas (= arm CE: router + norm gains + LoRA), MERGE the LoRA exactly into the expert weights
(W' = W + (alpha/r) * B @ A per expert), VERIFY the merged model reproduces CE's audited-slice BPB
(identity check) BEFORE any training, then FULL finetune (everything unfrozen, 8-bit Adam, LR 1e-5,
~200M tokens, evals every 25M at R=8 with telemetry). Reads: breaks the ~92.9% plateau => capacity
floor; same plateau => constraint price. Floor-probe runs full length (no early screen stop).

Usage: train_fprime.py <ce_full_ckpt.pt> <train_tokens> <tag> [eval_every]
"""
import os, sys, json, time, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES
from bitsandbytes.optim import Adam8bit
from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

CE_CKPT = sys.argv[1]; TRAIN_TOK = int(sys.argv[2]); TAG = sys.argv[3]
EVAL_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 25_000_000
LR = 1e-5; MB = int(os.environ.get("MB", "4"))          # full FT: grads on all params -> small MB
IMPOSE_BPB = 2.7507
OUT = "/workspace/olmoe-adapt/data"; CKPT = "/workspace/FLAME-MoE/results/ablations/adapt_ckpts"
D = json.load(open(f"{OUT}/bpb_slice_meta.json"))["divisor_D"]
BASE_BPB = 0.6727; IMP = 2.7507

model, tok = RES.load_model()
RES.enable_residency(R=8); RES.enable_grad_checkpointing(model)

# ---- reconstruct CE's trainable structure so the saved masters load into the right params ----
RES.freeze_all_but_router(model)
rp = RES.router_params(model)
norm_ps = RES.norm_params(model)
lora_ps = RES.add_lora(model, r=32, alpha=64)            # scale = alpha/r = 2.0
train_params = rp + norm_ps + lora_ps
ck = torch.load(CE_CKPT, map_location="cuda")
assert len(ck["masters"]) == len(train_params), f"masters {len(ck['masters'])} != params {len(train_params)}"
for p, m in zip(train_params, ck["masters"]):
    p.data.copy_(m.to("cuda").to(p.dtype))
print(f"[fprime] loaded CE deltas from {CE_CKPT} (seen={ck['seen']/1e6:.0f}M, {len(train_params)} tensors)", flush=True)

bpb_ids = torch.load(f"{OUT}/bpb_slice_ids.pt")
eval_sub = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, 256).long()].to("cuda").long()
E_experts = model.config.num_experts


def eval_bpb_telem():
    model.eval(); RES.reset_telem(); RES._CFG["collect_telem"] = True
    tot = n = 0
    with torch.no_grad():
        for i in range(eval_sub.shape[0]):
            x = eval_sub[i:i + 1]; out = model(x).logits.float()
            l = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)),
                                                  x[:, 1:].reshape(-1), reduction="sum")
            tot += l.item(); n += x[:, 1:].numel()
    RES._CFG["collect_telem"] = False; model.train()
    swap, ent = RES.telem_summary(E_experts)
    return (tot / n) / D, swap, ent


# ---- identity check: eval WITH LoRA active (== CE final), then merge, then eval WITH LoRA off ----
pre_bpb, _, _ = eval_bpb_telem()
scale = RES._LORA["scale"]
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, OlmoeExperts):
            for e in range(m.num_experts):
                m.gate_up_proj[e] += scale * (m.lora_gu_B[e] @ m.lora_gu_A[e])   # [2I,r]@[r,H]=[2I,H]
                m.down_proj[e]   += scale * (m.lora_dn_B[e] @ m.lora_dn_A[e])     # [H,r]@[r,I]=[H,I]
RES._LORA["scale"] = 0.0                                 # LoRA baked into experts -> forward adds 0
OlmoeExperts.forward = RES._orig_experts_forward         # restore plain expert forward (no wasted LoRA matmuls)
post_bpb, _, _ = eval_bpb_telem()
print(f"[fprime] identity check: pre-merge(LoRA active) BPB={pre_bpb:.4f}  post-merge(baked) BPB={post_bpb:.4f}  "
      f"|delta|={abs(pre_bpb-post_bpb):.5f}", flush=True)
if abs(pre_bpb - post_bpb) > 0.003:                     # ~0.5 sigma tolerance on the exact merge
    print(f"[ABORT] merge not identity ({abs(pre_bpb-post_bpb):.5f} > 0.003) — merge math wrong", flush=True)
    sys.exit(5)
parent_bpb = post_bpb
print(f"[fprime] merged parent BPB={parent_bpb:.4f} (rec {1-(parent_bpb-BASE_BPB)/(IMP-BASE_BPB):.4f}); "
      f"bar = beat by >2sigma (0.012). Now FULL finetune, 8-bit Adam lr={LR}.", flush=True)

# ---- full finetune: everything unfrozen EXCEPT the now-orphaned (merged-in) LoRA factors ----
lora_ids = set(id(p) for p in lora_ps)
for p in model.parameters():
    p.requires_grad = id(p) not in lora_ids
ft_params = [p for p in model.parameters() if p.requires_grad]
opt = Adam8bit(ft_params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
print(f"[fprime] full-FT trainable params: {sum(p.numel() for p in ft_params)/1e9:.2f}B", flush=True)
corpus = torch.load(f"{OUT}/finetune_ids.pt")
order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))
AUX_C, Z_C = 0.01, 0.001
model.train(); seen = step = pos = 0; hist = [{"tok": 0, "bpb": parent_bpb, "note": "merged parent"}]
t0 = time.time()
while seen < TRAIN_TOK:
    if pos + MB > corpus.shape[0]:
        pos = 0
    batch = corpus[order[pos:pos + MB]].to("cuda").long(); pos += MB
    out = model(batch, output_router_logits=True)
    logits = out.logits
    lm = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                           batch[:, 1:].reshape(-1))
    aux, z = RES.aux_z_from_router_logits(out.router_logits, batch.shape[0], batch.shape[1], 8)
    loss = lm + AUX_C * aux + Z_C * z
    if not torch.isfinite(loss):
        print(f"[ABORT] non-finite loss step {step}", flush=True); sys.exit(3)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ft_params, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    seen += batch.numel(); step += 1
    if step % 20 == 0:
        print(f"[step {step}] tok={seen/1e6:.1f}M lm={lm.item():.4f} {seen/(time.time()-t0)/1e3:.1f}k tok/s", flush=True)
    if seen // EVAL_EVERY > len([h for h in hist if h["tok"] > 0]):
        b, swap, ent = eval_bpb_telem()
        hist.append({"tok": seen, "bpb": b, "swap_rate": swap, "usage_entropy": ent})
        line = f"[eval] {TAG} arm=Fprime tok={seen/1e6:.0f}M BPB={b:.4f} swap={swap:.4f} ent={ent:.4f} (parent {parent_bpb:.4f})"
        print(line, flush=True); open(f"{OUT}/live_bakeoff.txt", "a").write(line + "\n")
        if b > IMPOSE_BPB:
            print(f"[ABORT] BPB {b:.4f} > impose {IMPOSE_BPB}", flush=True); sys.exit(4)

fb, fswap, fent = eval_bpb_telem()
json.dump({"arm": "Fprime", "tag": TAG, "lr": LR, "train_tokens": seen, "parent_bpb": parent_bpb,
           "final_bpb": fb, "final_swap": fswap, "final_entropy": fent, "divisor": D, "curve": hist},
          open(f"{OUT}/bakeoff_{TAG}.json", "w"), indent=1)
verdict = "BREAKS plateau (capacity floor)" if fb < parent_bpb - 0.012 else "SAME plateau (constraint price)"
open(f"{OUT}/live_bakeoff.txt", "a").write(f"[DONE] {TAG} arm=Fprime final_BPB={fb:.4f} parent={parent_bpb:.4f} -> {verdict}\n")
print(f"[DONE] arm=Fprime final BPB={fb:.4f} parent={parent_bpb:.4f} -> {verdict}", flush=True)
