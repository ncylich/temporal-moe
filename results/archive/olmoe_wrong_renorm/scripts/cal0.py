#!/usr/bin/env python3
"""Cal-0 (orch 0114): is arm C's norm-gain recovery just MOMENT MATCHING, computable closed-form with
ZERO training? Record per-channel input RMS at all 65 RMSNorm sites under free routing (A) vs greedy
R=8 (B); set g' = g * RMS_free/RMS_masked; eval R=8 canonical full-scan BPB (base router + matched
norms, no training); diagnose vs arm C's LEARNED gains. Writes olmoe_cal0.csv."""
import os, sys, json, math, numpy as np, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES
from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm
from safetensors.torch import load_file

N_STATS = int(sys.argv[1]) if len(sys.argv) > 1 else 512      # ~2M tok for the RMS stats
OUT = "/workspace/olmoe-adapt/data"
D = json.load(open(f"{OUT}/bpb_slice_meta.json"))["divisor_D"]
BASE, IMP = 0.6727, 2.7507

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

model, tok = RES.load_model(); RES.tag_layers(model)
norms = [m for m in model.modules() if isinstance(m, OlmoeRMSNorm)]
for i, m in enumerate(norms):
    m._cal_idx = i
base_g = [n.weight.detach().float().clone() for n in norms]
print(f"[cal0] {len(norms)} RMSNorm sites tagged; stats on {N_STATS} packs", flush=True)

ids = torch.load(f"{OUT}/bpb_slice_ids.pt").long()
stat_ids = ids[:N_STATS].to("cuda")
eval_sub = ids[torch.linspace(0, ids.shape[0] - 1, 256).long()].to("cuda")


def rms_pass(residency_on):
    ACC.clear(); CNT.clear()
    (RES.enable_residency(R=8) if residency_on else RES.disable_residency())
    REC["on"] = True
    with torch.no_grad():
        for i in range(stat_ids.shape[0]):
            model(stat_ids[i:i + 1])
    REC["on"] = False
    return {i: torch.sqrt(ACC[i] / CNT[i]) for i in ACC}


print("[cal0] pass A (free routing)...", flush=True); rms_free = rms_pass(False)
print("[cal0] pass B (greedy R=8)...", flush=True); rms_masked = rms_pass(True)

EPS = 1e-6                                                    # dead channels (always-0 input) -> keep base gain
ratio = []
for i in range(len(norms)):
    r = rms_free[i] / rms_masked[i].clamp(min=EPS)
    r = torch.where(rms_masked[i] < EPS, torch.ones_like(r), r)
    ratio.append(r)
gprime = [base_g[i] * ratio[i] for i in range(len(norms))]
gprime_clip = [base_g[i] * ratio[i].clamp(0.5, 2.0) for i in range(len(norms))]
print("[cal0] ratio stats: median=%.3f p1=%.3f p99=%.3f (per-channel RMS_free/RMS_masked)" %
      (float(torch.cat(ratio).median()), float(torch.cat(ratio).quantile(.01)), float(torch.cat(ratio).quantile(.99))), flush=True)


def set_norms(gs):
    for nmod, g in zip(norms, gs):
        nmod.weight.data.copy_(g.to(nmod.weight.dtype))


def eval_bpb(residency_on):
    (RES.enable_residency(R=8) if residency_on else RES.disable_residency())
    tot = ntk = 0
    with torch.no_grad():
        for i in range(eval_sub.shape[0]):
            x = eval_sub[i:i + 1]; o = model(x).logits.float()
            tot += torch.nn.functional.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)), x[:, 1:].reshape(-1), reduction="sum").item()
            ntk += x[:, 1:].numel()
    return (tot / ntk) / D


rec = lambda b: 1 - (b - BASE) / (IMP - BASE)
set_norms(gprime); bpb_m_r8 = eval_bpb(True); bpb_m_free = eval_bpb(False)
set_norms(gprime_clip); bpb_c_r8 = eval_bpb(True); bpb_c_free = eval_bpb(False)
set_norms(base_g)
print(f"[cal0] matched-norms @R8 BPB={bpb_m_r8:.4f} rec={rec(bpb_m_r8):.3f} | clipped @R8={bpb_c_r8:.4f} rec={rec(bpb_c_r8):.3f} | matched+free(sanity)={bpb_m_free:.4f}", flush=True)

# ---- diagnostic vs arm C's LEARNED gains ----
cC = load_file("/workspace/FLAME-MoE/results/ablations/adapt_ckpts/router_bake_C.safetensors")
gC = [cC[f"extra.{i}"].float() for i in range(len(norms))]      # arm C trained norm weights (same order)
sims = []
for i in range(len(norms)):
    dp = torch.log(gprime[i].cpu().clamp(min=1e-6) / base_g[i].cpu().clamp(min=1e-6))
    dc = torch.log(gC[i].clamp(min=1e-6) / base_g[i].cpu().clamp(min=1e-6))
    fin = torch.isfinite(dp) & torch.isfinite(dc)                # drop dead/degenerate channels
    if fin.sum() < 2 or dp[fin].norm() == 0 or dc[fin].norm() == 0:
        sims.append(float("nan")); continue
    cos = float(torch.nn.functional.cosine_similarity(dp[fin].unsqueeze(0), dc[fin].unsqueeze(0))[0])
    sims.append(cos)
# per-layer aggregate (norms grouped 4/layer + final)
os.makedirs("/workspace/FLAME-MoE/results/ablations", exist_ok=True)
CSV = "/workspace/FLAME-MoE/results/ablations/olmoe_cal0.csv"
lines = ["# Cal-0 (orch 0114): closed-form moment-matched norm gains g'=g*RMS_free/RMS_masked, ZERO training.",
         f"# stats {N_STATS} packs; eval 256 packs audited slice D={D:.4f}. Ladder: impose 2.7507 / A(router) 1.2825 / C(router+norms) 0.8505 / base 0.6727.",
         "row,BPB,recovery,note",
         f"matched_norms_R8,{bpb_m_r8:.4f},{rec(bpb_m_r8):.4f},closed-form g' + base router @R8 (NO training)",
         f"matched_clipped_R8,{bpb_c_r8:.4f},{rec(bpb_c_r8):.4f},g' clamped[0.5|2] @R8",
         f"matched_norms_free,{bpb_m_free:.4f},{rec(bpb_m_free):.4f},sanity: g_prime + FREE routing (should ~=base 0.6727)",
         f"matched_clipped_free,{bpb_c_free:.4f},{rec(bpb_c_free):.4f},sanity: clipped g_prime + FREE routing",
         "# per-norm cosine( log(g'/g) , log(gC/g) ) — direction agreement of closed-form vs arm C learned:",
         "norm_idx,cos_sim"]
for i, c in enumerate(sims):
    lines.append(f"{i},{c:.4f}")
lines.append(f"# mean cos-sim = {np.nanmean(sims):.4f}; median = {np.nanmedian(sims):.4f}; frac>0.5 = {np.mean([s>0.5 for s in sims if s==s]):.3f}")
open(CSV, "w").write("\n".join(lines) + "\n")
print(f"[cal0] wrote {CSV}", flush=True)
print(f"[cal0] VERDICT: closed-form moment-match @R8 = {bpb_m_r8:.4f} ({rec(bpb_m_r8)*100:.1f}%) vs arm C trained 0.8505 (91.4%). "
      f"mean gain-direction cos vs C = {np.nanmean(sims):.3f} ({np.mean([s>0.5 for s in sims if s==s])*100:.0f}% of sites aligned).", flush=True)
