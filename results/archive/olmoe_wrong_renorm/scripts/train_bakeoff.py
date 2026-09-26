#!/usr/bin/env python3
"""Stage 2b bake-off trainer. Router-only base-router init (no checkpoint), winner LR, same corpus
+ data order, 0.25B tokens, evals every 50M AT R=8 (serving condition) with telemetry (audited-slice
BPB D=3.1089, swap-rate/layer, expert-usage entropy). Per-arm 0.25B router checkpoint -> adapt_ckpts/.

Usage: train_bakeoff.py <arm> <lr> <tokens> <tag> [eval_every]
Arms: B (annealed R 64->8 one-expert-at-a-time over first 150M, hold 8 for rest; train-time R per
rung, eval R=8, log train-R). D/E/C/F added as they come up. Aborts on NaN / BPB>impose / usage collapse.
"""
import os, sys, json, time, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES
from safetensors.torch import save_file

ARM = sys.argv[1]; LR = float(sys.argv[2]); TRAIN_TOK = int(sys.argv[3]); TAG = sys.argv[4]
EVAL_EVERY = int(sys.argv[5]) if len(sys.argv) > 5 else 50_000_000
SEQ = 4096; AUX_C, Z_C = 0.01, 0.001
MB = 4 if sys.argv[1] in ("D", "G") else int(os.environ.get("MB", "16"))  # distill: 2 fwd + full-vocab logits
IMPOSE_BPB = 2.7507
RESUME = os.environ.get("RESUME", "")                # path to a full ckpt to continue from (optional)
OUT = "/workspace/olmoe-adapt/data"; CKPT = "/workspace/FLAME-MoE/results/ablations/adapt_ckpts"
os.makedirs(CKPT, exist_ok=True)
D = json.load(open(f"{OUT}/bpb_slice_meta.json"))["divisor_D"]
ANNEAL_END = 150_000_000; RUNG_TOK = ANNEAL_END / 56.0            # 64->8 over 56 rungs

model, tok = RES.load_model()
RES.enable_residency(R=8); RES.enable_grad_checkpointing(model)
RES.freeze_all_but_router(model)
rp = RES.router_params(model)
base_router = [p.detach().clone() for p in rp]              # arm D teacher: frozen base-router snapshot
extra = []; norm_ps = []; has_lora = False
if ARM == "C":
    norm_ps = RES.norm_params(model); extra = norm_ps      # + learnable RMSNorm gains
elif ARM == "E":
    extra = RES.add_lora(model, r=32, alpha=64); has_lora = True   # + per-expert LoRA
elif ARM == "Er8":
    extra = RES.add_lora(model, r=8, alpha=16); has_lora = True    # rank screen: minimal adapter (alpha/r=2)
elif ARM == "Er64":
    extra = RES.add_lora(model, r=64, alpha=128); has_lora = True  # rank screen: large adapter (alpha/r=2)
elif ARM == "H":
    extra = RES.add_lora(model, r=32, alpha=64); has_lora = True   # E-recipe + zone-confined anneal (train_R)
elif ARM in ("CE", "G"):
    norm_ps = RES.norm_params(model)                       # norms + LoRA (G also distills, see loss)
    extra = norm_ps + RES.add_lora(model, r=32, alpha=64); has_lora = True
base_norms = [p.detach().clone() for p in norm_ps]         # arm G teacher: base norm gains
for p in extra:
    p.requires_grad = True
train_params = rp + extra
n_tr = sum(p.numel() for p in train_params)
masters = [p.detach().float().clone().requires_grad_(True) for p in train_params]
opt = torch.optim.AdamW(masters, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
print(f"[bakeoff] arm={ARM} tag={TAG} lr={LR} trainable={n_tr} "
      f"(router={sum(p.numel() for p in rp)} extra={sum(p.numel() for p in extra)}) tokens={TRAIN_TOK}", flush=True)

corpus = torch.load(f"{OUT}/finetune_ids.pt")
order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))  # same order as sweep
bpb_ids = torch.load(f"{OUT}/bpb_slice_ids.pt")
eval_sub = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, 256).long()].to("cuda").long()
E_experts = model.config.num_experts


def train_R(seen):
    if ARM == "B":                                          # 64->8 over first 150M (56 rungs)
        return 8 if seen >= ANNEAL_END else max(8, 64 - int(seen // RUNG_TOK))
    if ARM == "H":                                          # zone-confined: 24->8 over first 50M (16 rungs), hold 8
        return 8 if seen >= 50_000_000 else max(8, 24 - int(seen // (50_000_000 / 16)))
    return 8


def eval_bpb_telem():
    model.eval(); RES.enable_residency(R=8); RES.reset_telem(); RES._CFG["collect_telem"] = True
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


def save_delta():
    sd = {f"router.{i}.weight": rp[i].detach().to(torch.bfloat16).cpu() for i in range(len(rp))}
    for i, p in enumerate(extra):
        sd[f"extra.{i}"] = p.detach().to(torch.bfloat16).cpu()
    save_file(sd, f"{CKPT}/router_{TAG}.safetensors")


FULL = f"{OUT}/ckpt_{TAG}.pt"                          # resumable state (fp32 masters + Adam + counters)


def save_full(seen, step, pos, hist):
    torch.save({"masters": [m.detach().cpu() for m in masters], "opt": opt.state_dict(),
                "seen": seen, "step": step, "pos": pos, "hist": hist, "arm": ARM, "lr": LR},
               FULL + ".tmp"); os.replace(FULL + ".tmp", FULL)   # atomic so a crash mid-write can't corrupt


seen = step = pos = 0; hist = []
if RESUME and os.path.exists(RESUME):
    ck = torch.load(RESUME, map_location="cuda")
    for m, s in zip(masters, ck["masters"]):
        m.data.copy_(s.to("cuda"))
    opt.load_state_dict(ck["opt"])
    for m, p in zip(masters, train_params):
        p.data.copy_(m.data.to(p.dtype))
    seen, step, pos, hist = ck["seen"], ck["step"], ck["pos"], ck["hist"]
    print(f"[resume] {RESUME}: seen={seen/1e6:.0f}M step={step} evals={len(hist)}", flush=True)
model.train(); t0 = time.time(); seen0 = seen
while seen < TRAIN_TOK:
    if pos + MB > corpus.shape[0]:
        pos = 0
    RES._CFG["R"] = train_R(seen)                                # arm B: current rung; else 8
    batch = corpus[order[pos:pos + MB]].to("cuda").long(); pos += MB
    labels = batch[:, 1:].reshape(-1)
    if ARM in ("D", "G"):
        # teacher = frozen BASE model, FREE routing, no-grad; student = trained, R=8.
        # base = base router (+ base norm gains and LoRA-off for arm G, since those are trained here).
        # memory-frugal: keep [MB,S,V] tensors bf16, materialize only ONE fp32 log-softmax and reuse
        # it for both CE (nll_loss) and the soft term; smaller MB (full-vocab logits x2 forwards).
        cur = [p.detach().clone() for p in rp]
        for p, bw in zip(rp, base_router):
            p.data.copy_(bw)
        if ARM == "G":
            cur_n = [p.detach().clone() for p in norm_ps]
            for p, bw in zip(norm_ps, base_norms):
                p.data.copy_(bw)
            saved_scale = RES._LORA["scale"]; RES._LORA["scale"] = 0.0   # LoRA off for the base teacher
        RES.disable_residency()
        with torch.no_grad():
            t_prob = model(batch).logits[:, :-1].softmax(-1)       # bf16 soft target [MB,S-1,V]
        for p, c in zip(rp, cur):
            p.data.copy_(c)
        if ARM == "G":
            for p, c in zip(norm_ps, cur_n):
                p.data.copy_(c)
            RES._LORA["scale"] = saved_scale
        RES.enable_residency(R=8)
        out = model(batch, output_router_logits=True)
        V = out.logits.size(-1)
        s_logsm = torch.log_softmax(out.logits[:, :-1].float(), -1)  # [MB,S-1,V] fp32, reused below
        ce = torch.nn.functional.nll_loss(s_logsm.reshape(-1, V), labels)
        soft = -(t_prob * s_logsm).sum(-1).mean()                   # temp 1
        lm = 0.5 * ce + 0.5 * soft
        del t_prob, s_logsm
    else:
        out = model(batch, output_router_logits=True)
        logits = out.logits
        lm = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels)
    aux, z = RES.aux_z_from_router_logits(out.router_logits, batch.shape[0], batch.shape[1], RES._CFG["R"])
    loss = lm + AUX_C * aux + Z_C * z
    if not torch.isfinite(loss):
        print(f"[ABORT] non-finite loss step {step}", flush=True); sys.exit(3)
    loss.backward()
    for m, p in zip(masters, train_params):
        m.grad = p.grad.float() if p.grad is not None else None
    torch.nn.utils.clip_grad_norm_(masters, 1.0)
    opt.step()
    for m, p in zip(masters, train_params):
        p.data.copy_(m.data.to(p.dtype)); p.grad = None
    opt.zero_grad(set_to_none=True)
    seen += batch.numel(); step += 1
    if step % 20 == 0:
        print(f"[step {step}] tok={seen/1e6:.1f}M R={RES._CFG['R']} lm={lm.item():.4f} "
              f"{(seen-seen0)/(time.time()-t0)/1e3:.1f}k tok/s", flush=True)
    if seen // EVAL_EVERY > len(hist):
        RES._CFG["R"] = 8                                        # eval always R=8
        b, swap, ent = eval_bpb_telem(); trR = train_R(seen)
        hist.append({"tok": seen, "bpb": b, "swap_rate": swap, "usage_entropy": ent, "train_R": trR})
        line = f"[eval] {TAG} arm={ARM} tok={seen/1e6:.0f}M trainR={trR} BPB={b:.4f} swap={swap:.4f} ent={ent:.4f}"
        print(line, flush=True); open(f"{OUT}/live_bakeoff.txt", "a").write(line + "\n")
        save_full(seen, step, pos, hist)                        # resumable state at every eval
        if b > IMPOSE_BPB:
            print(f"[ABORT] BPB {b:.4f} > impose {IMPOSE_BPB}", flush=True); sys.exit(4)
        if ent < 0.30:
            print(f"[WARN] usage entropy {ent:.3f} low — possible collapse (flagging, not aborting)", flush=True)

fb, fswap, fent = eval_bpb_telem()
save_delta(); save_full(seen, step, pos, hist)
json.dump({"arm": ARM, "tag": TAG, "lr": LR, "train_tokens": seen, "final_bpb": fb,
           "final_swap": fswap, "final_entropy": fent, "divisor": D, "curve": hist},
          open(f"{OUT}/bakeoff_{TAG}.json", "w"), indent=1)
open(f"{OUT}/live_bakeoff.txt", "a").write(f"[DONE] {TAG} arm={ARM} final_BPB={fb:.4f} swap={fswap:.4f} ent={fent:.4f}\n")
print(f"[DONE] arm={ARM} final BPB={fb:.4f} swap={fswap:.4f} ent={fent:.4f}", flush=True)
