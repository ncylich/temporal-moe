#!/usr/bin/env python3
"""Stage 2 router-only finetune under the hard residency constraint (R=k=8 of 64), constraint on
from step 0. Trainable = the 16 OlmoeTopKRouter.weight linears (~2.1M params); everything else
requires_grad=False. bf16 compute + fp32 master for the router. Loss = LM CE (through the resident-
masked softmax) + aux(0.01) + z-loss(0.001) on the MASKED distribution. Packs at 4096.

Usage: train_router.py <lr> <train_tokens> <tag> [eval_every_tokens]
Evals audited-slice BPB (D=3.1089) at the end (and every eval_every_tokens if given), writing a
per-run json + saving the router-only delta to data/router_<tag>.safetensors. Aborts on NaN or if
adapted BPB exceeds the impose level (2.7507).
"""
import os, sys, json, math, time, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES
from safetensors.torch import save_file

LR = float(sys.argv[1]); TRAIN_TOK = int(sys.argv[2]); TAG = sys.argv[3]
EVAL_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SEQ = 4096; MB = int(os.environ.get("MB", "2")); AUX_C, Z_C = 0.01, 0.001
IMPOSE_BPB = 2.7507
OUT = "/workspace/olmoe-adapt/data"
meta = json.load(open(f"{OUT}/bpb_slice_meta.json")); D = meta["divisor_D"]
dev = "cuda"

model, tok = RES.load_model()
RES.enable_residency(R=8)
RES.enable_grad_checkpointing(model)                 # recompute activations -> fits large MB
n_tr = RES.freeze_all_but_router(model)
rp = RES.router_params(model)
masters = [p.detach().float().clone().requires_grad_(True) for p in rp]
opt = torch.optim.AdamW(masters, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
print(f"[train] tag={TAG} lr={LR} trainable={n_tr} (router only) tokens={TRAIN_TOK}", flush=True)

corpus = torch.load(f"{OUT}/finetune_ids.pt")                      # [Nseq,4096] int32
g = torch.Generator().manual_seed(0)
order = torch.randperm(corpus.shape[0], generator=g)
bpb_ids = torch.load(f"{OUT}/bpb_slice_ids.pt")
eval_idx = torch.linspace(0, bpb_ids.shape[0] - 1, 128).long()
eval_sub = bpb_ids[eval_idx].to(dev).long()


def eval_bpb():
    model.eval(); RES.enable_residency(R=8)
    tot = n = 0
    with torch.no_grad():
        for i in range(eval_sub.shape[0]):
            x = eval_sub[i:i + 1]; out = model(x).logits.float()
            l = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)),
                                                  x[:, 1:].reshape(-1), reduction="sum")
            tot += l.item(); n += x[:, 1:].numel()
    model.train()
    return (tot / n) / D


def save_delta():
    sd = {f"router.{i}.weight": rp[i].detach().to(torch.bfloat16).cpu() for i in range(len(rp))}
    save_file(sd, f"{OUT}/router_{TAG}.safetensors")


base_bpb = None
model.train()
seen = 0; step = 0; t0 = time.time(); pos = 0; hist = []
while seen < TRAIN_TOK:
    if pos + MB > corpus.shape[0]:
        pos = 0
    batch = corpus[order[pos:pos + MB]].to(dev).long(); pos += MB
    out = model(batch, output_router_logits=True)
    logits = out.logits
    lm = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                           batch[:, 1:].reshape(-1))
    aux, z = RES.aux_z_from_router_logits(out.router_logits, batch.shape[0], batch.shape[1], 8)
    loss = lm + AUX_C * aux + Z_C * z
    if not torch.isfinite(loss):
        print(f"[ABORT] non-finite loss at step {step}: lm={lm}", flush=True); sys.exit(3)
    loss.backward()
    for m, p in zip(masters, rp):
        m.grad = p.grad.float() if p.grad is not None else None
    torch.nn.utils.clip_grad_norm_(masters, 1.0)
    opt.step()
    for m, p in zip(masters, rp):
        p.data.copy_(m.data.to(p.dtype))
    opt.zero_grad(set_to_none=True)
    for p in rp:
        p.grad = None
    seen += batch.numel(); step += 1
    if step % 20 == 0:
        tps = seen / (time.time() - t0)
        print(f"[step {step}] tok={seen/1e6:.1f}M lm={lm.item():.4f} aux={float(aux):.4f} "
              f"z={float(z):.4f} {tps/1e3:.1f}k tok/s", flush=True)
    if EVAL_EVERY and seen // EVAL_EVERY > len(hist):
        b = eval_bpb(); hist.append((seen, b))
        line = f"[eval] {TAG} lr={LR} tok={seen/1e6:.0f}M adapted_BPB={b:.4f}"
        print(line, flush=True)
        open(f"{OUT}/live_sweep.txt", "a").write(line + "\n")   # live curve, runner-independent
        if b > IMPOSE_BPB:
            print(f"[ABORT] adapted BPB {b:.4f} exceeds impose {IMPOSE_BPB}", flush=True); sys.exit(4)

final_bpb = eval_bpb()
save_delta()
json.dump({"tag": TAG, "lr": LR, "train_tokens": seen, "final_bpb": final_bpb,
           "divisor": D, "curve": hist, "trainable_params": n_tr},
          open(f"{OUT}/train_{TAG}.json", "w"), indent=1)
open(f"{OUT}/live_sweep.txt", "a").write(f"[DONE] {TAG} lr={LR} final_BPB={final_bpb:.4f}\n")
print(f"[DONE] tag={TAG} lr={LR} final adapted BPB={final_bpb:.4f}", flush=True)
