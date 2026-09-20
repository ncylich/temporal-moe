#!/usr/bin/env python3
"""Cal-2 (orch 0132): INIT-AXIS screen. Arm-C surface (router + all 65 RMSNorm gains, both trainable),
lr 3e-4, 50M tokens, eval @R=8 every 10M -- but the norm gains are INITIALIZED from the CLIPPED
closed-form g' (Cal-0's 2.0965-BPB moment-matched start) instead of base. Router starts from base.
Pre-registered bar: beat arm C's like-for-like curve (C@50M=0.8791) by >2sigma(0.012) -> promote 250M;
else the null CLOSES the init axis. Per eval also logs the mechanism telemetry:
  cos_prime = cos( log(g_t/g_base), log(g'/g_base) )   -- does training RETAIN the calibrated direction?
  cos_C     = cos( log(g_t/g_base), log(g_C/g_base) )   -- or UNDO it toward arm C's learned basin?
Writes olmoe_cal2.csv (BPB curve + both cosine trajectories) + prints the 3-way verdict.
Usage: train_cal2.py [lr] [tokens] [eval_every] [n_stats]"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES
from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm
from safetensors.torch import load_file, save_file

LR = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-4
TRAIN_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000_000
EVAL_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000
N_STATS = int(sys.argv[4]) if len(sys.argv) > 4 else 512
SEQ = 4096; MB = int(os.environ.get("MB", "16")); AUX_C, Z_C = 0.01, 0.001
IMPOSE_BPB = 2.7507; BASE, IMP = 0.6727, 2.7507; C_AT_50M = 0.8791
OUT = "/workspace/olmoe-adapt/data"; CKPT = "/workspace/FLAME-MoE/results/ablations/adapt_ckpts"
D = json.load(open(f"{OUT}/bpb_slice_meta.json"))["divisor_D"]
rec = lambda b: 1 - (b - BASE) / (IMP - BASE)

# ---- model + arm-C surface (router + norm gains both trainable) ----
model, tok = RES.load_model()
RES.enable_residency(R=8); RES.enable_grad_checkpointing(model)
RES.freeze_all_but_router(model)
rp = RES.router_params(model)
norm_ps = RES.norm_params(model)                                 # 65 RMSNorm gains, module order
norms = [m for m in model.modules() if isinstance(m, OlmoeRMSNorm)]
assert len(norms) == len(norm_ps) == 65, (len(norms), len(norm_ps))
base_g = [n.weight.detach().float().clone().cpu() for n in norms]   # g_base (before g' init)
gC = load_file(f"{CKPT}/router_bake_C.safetensors")                 # arm C learned gains (extra.{i})
g_C = [gC[f"extra.{i}"].float().cpu() for i in range(len(norms))]

# ---- Cal-0 RMS passes -> clipped closed-form g' = g * clamp(RMS_free/RMS_masked, 0.5, 2.0) ----
ACC, CNT, REC = {}, {}, {"on": False}
_orig_norm = OlmoeRMSNorm.forward
def _rec_forward(self, x):
    if REC["on"]:
        i = self._cal_idx; xf = x.detach().float().reshape(-1, x.shape[-1])
        s = xf.pow(2).sum(0)
        ACC[i] = s if i not in ACC else ACC[i] + s
        CNT[i] = CNT.get(i, 0) + xf.shape[0]
    return _orig_norm(self, x)
OlmoeRMSNorm.forward = _rec_forward
for i, m in enumerate(norms):
    m._cal_idx = i
ids = torch.load(f"{OUT}/bpb_slice_ids.pt").long()
stat_ids = ids[:N_STATS].to("cuda")
def rms_pass(residency_on):
    ACC.clear(); CNT.clear()
    (RES.enable_residency(R=8) if residency_on else RES.disable_residency())
    REC["on"] = True
    with torch.no_grad():
        for i in range(stat_ids.shape[0]):
            model(stat_ids[i:i + 1])
    REC["on"] = False
    return {i: torch.sqrt(ACC[i] / CNT[i]) for i in ACC}
print(f"[cal2] computing g' from {N_STATS}-pack RMS passes (free vs R=8)...", flush=True)
rms_free = rms_pass(False); rms_masked = rms_pass(True)
EPS = 1e-6
gprime_clip = []
for i in range(len(norms)):
    r = rms_free[i] / rms_masked[i].clamp(min=EPS)
    r = torch.where(rms_masked[i] < EPS, torch.ones_like(r), r).clamp(0.5, 2.0)
    gprime_clip.append((base_g[i].to("cuda") * r).float().cpu())
OlmoeRMSNorm.forward = _orig_norm                                # unpatch: no recording during training
torch.save({"gprime_clip": gprime_clip, "base_g": base_g}, f"{OUT}/cal2_gprime.pt")

# ---- INIT: norm gains <- g'_clip ; router stays base ----
def set_norms(gs):
    for nmod, g in zip(norms, gs):
        nmod.weight.data.copy_(g.to(nmod.weight.dtype).to(nmod.weight.device))
set_norms(gprime_clip)

# ---- trainable set: router + norm gains (arm-C surface) ----
extra = norm_ps
for p in extra:
    p.requires_grad = True
train_params = rp + extra
masters = [p.detach().float().clone().requires_grad_(True) for p in train_params]
opt = torch.optim.AdamW(masters, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
n_tr = sum(p.numel() for p in train_params)
print(f"[cal2] arm-C surface trainable={n_tr} (router={sum(p.numel() for p in rp)} norms={sum(p.numel() for p in extra)}) "
      f"lr={LR} tok={TRAIN_TOK} eval_every={EVAL_EVERY}", flush=True)

corpus = torch.load(f"{OUT}/finetune_ids.pt")
order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))  # same order as bake-off
bpb_ids = torch.load(f"{OUT}/bpb_slice_ids.pt")
eval_sub = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, 256).long()].to("cuda").long()
E_experts = model.config.num_experts


def eval_bpb_telem():
    model.eval(); RES.enable_residency(R=8); RES.reset_telem(); RES._CFG["collect_telem"] = True
    tot = n = 0
    with torch.no_grad():
        for i in range(eval_sub.shape[0]):
            x = eval_sub[i:i + 1]; out = model(x).logits.float()
            tot += torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)),
                                                      x[:, 1:].reshape(-1), reduction="sum").item()
            n += x[:, 1:].numel()
    RES._CFG["collect_telem"] = False; model.train()
    swap, ent = RES.telem_summary(E_experts)
    return (tot / n) / D, swap, ent


def gain_cosines():
    """global direction agreement of the CURRENT gains vs g' and vs g_C (concatenated over 65 sites)."""
    g_t = [nmod.weight.detach().float().cpu() for nmod in norms]
    dt, dp, dc = [], [], []
    for i in range(len(norms)):
        b = base_g[i].clamp(min=1e-6)
        dt.append(torch.log(g_t[i].clamp(min=1e-6) / b))
        dp.append(torch.log(gprime_clip[i].clamp(min=1e-6) / b))
        dc.append(torch.log(g_C[i].clamp(min=1e-6) / b))
    dt, dp, dc = torch.cat(dt), torch.cat(dp), torch.cat(dc)
    fin = torch.isfinite(dt) & torch.isfinite(dp) & torch.isfinite(dc)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a[fin][None], b[fin][None])[0])
    return cos(dt, dp), cos(dt, dc)


hist = []
# eval 0: the g'-init start itself (sanity: should reproduce Cal-0 clipped ~2.0965; cos_prime=1.0)
b0, sw0, en0 = eval_bpb_telem(); cp0, cc0 = gain_cosines()
hist.append({"tok": 0, "bpb": b0, "swap_rate": sw0, "usage_entropy": en0, "cos_prime": cp0, "cos_C": cc0})
print(f"[cal2 eval] tok=0M BPB={b0:.4f} (g'-init sanity vs Cal-0 clipped 2.0965) rec={rec(b0):.3f} "
      f"cos_prime={cp0:.4f} cos_C={cc0:.4f}", flush=True)

seen = step = pos = 0; t0 = time.time()
model.train()
while seen < TRAIN_TOK:
    if pos + MB > corpus.shape[0]:
        pos = 0
    batch = corpus[order[pos:pos + MB]].to("cuda").long(); pos += MB
    labels = batch[:, 1:].reshape(-1)
    out = model(batch, output_router_logits=True)
    logits = out.logits
    lm = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels)
    aux, z = RES.aux_z_from_router_logits(out.router_logits, batch.shape[0], batch.shape[1], 8)
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
        print(f"[step {step}] tok={seen/1e6:.1f}M lm={lm.item():.4f} "
              f"{seen/(time.time()-t0)/1e3:.1f}k tok/s", flush=True)
    if seen // EVAL_EVERY > len([h for h in hist if h['tok'] > 0]):
        b, swap, ent = eval_bpb_telem(); cp, cc = gain_cosines()
        hist.append({"tok": seen, "bpb": b, "swap_rate": swap, "usage_entropy": ent, "cos_prime": cp, "cos_C": cc})
        print(f"[cal2 eval] tok={seen/1e6:.0f}M BPB={b:.4f} rec={rec(b):.3f} vs C@50M={C_AT_50M} "
              f"swap={swap:.4f} ent={ent:.4f} cos_prime={cp:.4f} cos_C={cc:.4f}", flush=True)
        if b > IMPOSE_BPB:
            print(f"[ABORT] BPB {b:.4f} > impose {IMPOSE_BPB}", flush=True); sys.exit(4)

fb, fswap, fent = eval_bpb_telem(); fcp, fcc = gain_cosines()
if hist[-1]["tok"] != seen:
    hist.append({"tok": seen, "bpb": fb, "swap_rate": fswap, "usage_entropy": fent, "cos_prime": fcp, "cos_C": fcc})
save_file({f"router.{i}.weight": rp[i].detach().to(torch.bfloat16).cpu() for i in range(len(rp))} |
          {f"extra.{i}": extra[i].detach().to(torch.bfloat16).cpu() for i in range(len(extra))},
          f"{CKPT}/router_cal2.safetensors")

# ---- verdict + CSV ----
delta = C_AT_50M - fb                                            # >0 means Cal-2 beats arm C (lower BPB)
sig2 = 0.012
if delta > sig2:
    verdict = f"WIN: g'-init reaches BPB {fb:.4f} < C@50M {C_AT_50M} by {delta:.4f} (>2sigma {sig2}) -> promote 250M"
elif abs(delta) <= sig2:
    basin = "equal-quality DIFFERENT basin" if fcp > 0.3 and fcp > fcc else "equal-quality, UNDONE toward arm C single basin"
    verdict = f"EQUAL ({delta:+.4f}, within 2sigma {sig2}): {basin}; init axis: no BPB gain"
else:
    verdict = f"NULL: g'-init WORSE by {-delta:.4f} -> init axis CLOSED (arm C's from-base start is not improved)"
retain = "RETAINS g' direction" if fcp > fcc and fcp > 0.3 else "UNDOES g' toward arm C (g_C)"
CSV = "/workspace/FLAME-MoE/results/ablations/olmoe_cal2.csv"
lines = ["# Cal-2 (orch 0132): INIT-AXIS screen. Arm-C surface (router+65 norm gains both trainable), lr %.0e, %dM tok," % (LR, TRAIN_TOK//1_000_000),
         "# norm gains INITIALIZED from clipped closed-form g' (Cal-0 2.0965-BPB start); router from base. Eval @R=8, D=%.4f." % D,
         "# Bar: beat arm C like-for-like C@50M=%.4f by >2sigma(%.3f). Ladder: base %.4f / arm C 0.8505 / impose %.4f." % (C_AT_50M, sig2, BASE, IMP),
         "# cos_prime=cos(log(g_t/g_base),log(g'/g_base)) [retain g'?]; cos_C=cos(log(g_t/g_base),log(g_C/g_base)) [undo toward arm C?]",
         "tok_M,BPB,recovery,swap_rate,usage_entropy,cos_prime,cos_C"]
for h in hist:
    lines.append(f"{h['tok']/1e6:.0f},{h['bpb']:.4f},{rec(h['bpb']):.4f},{h['swap_rate']:.4f},{h['usage_entropy']:.4f},{h['cos_prime']:.4f},{h['cos_C']:.4f}")
lines.append(f"# FINAL @ {seen/1e6:.0f}M: BPB={fb:.4f} rec={rec(fb):.3f} cos_prime={fcp:.4f} cos_C={fcc:.4f} ({retain})")
lines.append(f"# VERDICT: {verdict}")
open(CSV, "w").write("\n".join(lines) + "\n")
json.dump({"lr": LR, "train_tokens": seen, "final_bpb": fb, "C_at_50M": C_AT_50M, "delta_vs_C": delta,
           "final_cos_prime": fcp, "final_cos_C": fcc, "divisor": D, "curve": hist, "verdict": verdict},
          open(f"{OUT}/cal2.json", "w"), indent=1)
print(f"[cal2] wrote {CSV}", flush=True)
print(f"[cal2] {verdict}", flush=True)
print(f"[cal2] mechanism: final cos_prime={fcp:.4f} cos_C={fcc:.4f} -> {retain}", flush=True)
