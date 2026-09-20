#!/usr/bin/env python3
"""O-2 reward->BPB transfer (orch 0106): eval the FROZEN base at R=8 on the audited slice under 5
schedules — static-best-8, greedy-scan, MinFlow m=1/2/4 — ALL windowed at TW=256 (incl. greedy, so it
is its own baseline, differing from the canonical 2.7507). Plus the free top-8 anchor (= base 0.6727).
Deliver BPB per schedule + captured-mass<->BPB fit + schedulability headline (MinFlow m1 BPB vs
windowed-greedy BPB) -> olmoe_minflow_bpb.csv.

Schedules are built from the captured base-softmax reward fields (data/oseries), then forced into the
model via RES.set_forced; BPB is measured on the same packs. Usage: oseries_o2.py <n_packs> [tw]
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts"); sys.path.insert(0, "/workspace/FLAME-MoE")
import olmoe_residency as RES
from temporal.temporal_router import compute_resident_mask
from ortools.graph.python import min_cost_flow

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
TW = int(sys.argv[2]) if len(sys.argv) > 2 else 256
C = 16; K = 8; SCALE = 1_000_000
DD = "/workspace/olmoe-adapt/data/oseries"; meta = json.load(open(f"{DD}/meta.json"))
NS, L, T, E, TOPK = meta["n_seq"], meta["L"], meta["seq"], meta["E"], meta["topk"]
D = json.load(open("/workspace/olmoe-adapt/data/bpb_slice_meta.json"))["divisor_D"]
idx = np.memmap(f"{DD}/idx.u8", np.uint8, "r", shape=(NS, L, T, TOPK))
val = np.memmap(f"{DD}/val.f16", np.float16, "r", shape=(NS, L, T, TOPK))
n = min(N, NS); NW = T // TW; MS = [1, 2, 4]
print(f"[o2] {n} packs, TW={TW} ({NW} win), schedules: static, greedy, MinFlow m{MS}", flush=True)


def dense(s, l):
    m = np.zeros((T, E)); np.put_along_axis(m, idx[s, l].astype(np.int64), val[s, l].astype(np.float64), 1); return m


def _solve_mask(win, m, Cx):                                  # EXACT (no parking): C=Cx candidates/token
    cand = np.argsort(-win, 1)[:, :Cx]; cm = np.take_along_axis(win, cand, 1)
    IN = lambda t, j: t * Cx + j; OUT = lambda t, j: Cx * TW + t * Cx + j
    HIN = lambda t: 2 * Cx * TW + t; HOUT = lambda t: 2 * Cx * TW + TW + t
    S = 2 * Cx * TW + 2 * TW; Ksink = S + 1; jr = np.arange(Cx); T1 = TW - 1
    tl, hd, cp, co = [], [], [], []
    def a(t, h, c, o): tl.append(np.asarray(t)); hd.append(np.asarray(h)); cp.append(np.asarray(c, np.int64)); co.append(np.asarray(o, np.int64))
    tt = np.repeat(np.arange(TW), Cx); jj = np.tile(jr, TW)
    a(tt * Cx + jj, Cx * TW + tt * Cx + jj, np.ones(Cx * TW), -np.rint(cm.reshape(-1) * SCALE))  # reward
    a(np.full(Cx, S), jr, np.ones(Cx), np.zeros(Cx))                                             # cold
    eq = cand[:-1, :, None] == cand[1:, None, :]; tp, ja, jb = np.nonzero(eq)
    a(OUT(tp, ja), IN(tp + 1, jb), np.ones(len(tp)), np.zeros(len(tp)))                          # stay
    te = np.repeat(np.arange(T1), Cx); je = np.tile(jr, T1)
    a(OUT(te, je), HIN(te + 1), np.ones(Cx * T1), np.zeros(Cx * T1))                             # evict
    ht = np.arange(1, TW); a(HIN(ht), HOUT(ht), np.full(T1, m), np.zeros(T1))                    # hub cap m
    a(HOUT(te + 1), IN(te + 1, je), np.ones(Cx * T1), np.zeros(Cx * T1))                         # admit
    a(OUT(np.full(Cx, T1), jr), np.full(Cx, Ksink), np.ones(Cx), np.zeros(Cx))                   # drain
    g = min_cost_flow.SimpleMinCostFlow(); nr = Cx * TW
    g.add_arcs_with_capacity_and_unit_cost(np.concatenate(tl), np.concatenate(hd), np.concatenate(cp), np.concatenate(co))
    g.set_node_supply(int(S), K); g.set_node_supply(int(Ksink), -K)
    if g.solve() != g.OPTIMAL:
        return None
    fl = np.array([g.flow(i) for i in range(nr)]).reshape(TW, Cx)      # reward-arc flows -> exactly 8/token
    mask = np.zeros((TW, E), bool)
    for t in range(TW):
        mask[t, cand[t][fl[t] > 0]] = True
    return mask


def minflow_mask(win, m):                                     # exact 8-resident mask; grow C until feasible
    for Cx in (32, 48, 64):
        mk = _solve_mask(win, m, Cx)
        if mk is not None:
            return mk
    # last resort: greedy (never happens for C=64 which includes all experts)
    lg = torch.from_numpy(np.where(win > 0, np.log(win + 1e-12), -1e9)).float().unsqueeze(1)
    return compute_resident_mask(lg, K, evict="min_logit").squeeze(1).numpy()


def sched_masks(s):                                           # -> dict name -> list-of-[T,E] per layer
    out = {k: [] for k in ["static", "greedy"] + [f"minflow_m{m}" for m in MS]}
    for l in range(L):
        mass = dense(s, l)
        st = np.zeros((T, E), bool); gr = np.zeros((T, E), bool); mf = {m: np.zeros((T, E), bool) for m in MS}
        for w in range(NW):
            sl = slice(w * TW, (w + 1) * TW); win = mass[sl]
            best8 = np.argsort(-win.sum(0))[:K]; st[sl][:, best8] = True
            lg = torch.from_numpy(np.where(win > 0, np.log(win + 1e-12), -1e9)).float().unsqueeze(1)
            gr[sl] = compute_resident_mask(lg, K, evict="min_logit").squeeze(1).numpy()
            for m in MS:
                mf[m][sl] = minflow_mask(win, m)
        out["static"].append(st); out["greedy"].append(gr)
        for m in MS:
            out[f"minflow_m{m}"].append(mf[m])
    return out


model, tok = RES.load_model(); RES.tag_layers(model); RES.enable_residency(R=8)
ids = torch.load("/workspace/olmoe-adapt/data/bpb_slice_ids.pt")[:n].long()


def bpb_forced(x, masks_or_none):
    RES.set_forced(masks_or_none)
    with torch.no_grad():
        out = model(x).logits.float()
        ce = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
    return ce.item(), x[:, 1:].numel()


names = ["free_top8", "static", "greedy", "minflow_m1", "minflow_m2", "minflow_m4"]
totce = {k: 0.0 for k in names}; ntok = 0
for si in range(n):
    x = ids[si:si + 1].to("cuda")
    RES.disable_residency(); c, nt = bpb_forced(x, None); totce["free_top8"] += c   # free anchor (no residency)
    RES.enable_residency(R=8)
    sm = sched_masks(si)
    for k in names[1:]:
        masks = [torch.from_numpy(sm[k][l]).unsqueeze(1) for l in range(L)]         # [T,1,E] per layer
        c, _ = bpb_forced(x, masks); totce[k] += c
    ntok += nt
    print(f"[o2] pack {si+1}/{n} done", flush=True)
RES.set_forced(None)

bpb = {k: totce[k] / ntok / D for k in names}
# captured-mass anchors for the fit (per-token mean served mass): free=top8 bound; others from schedule masks reused
os.makedirs("/workspace/FLAME-MoE/results/ablations", exist_ok=True)
CSV = "/workspace/FLAME-MoE/results/ablations/olmoe_minflow_bpb.csv"
lines = ["# O-2 reward->BPB transfer (orch 0106). Frozen base @ R=8, audited slice (D=%.4f), %d packs, TW=%d." % (D, n, TW),
         "# ALL schedules windowed at TW=256 incl. greedy (its own baseline, != canonical impose 2.7507).",
         "# free_top8 = base free routing anchor (no residency). Lower BPB = better.",
         "schedule,BPB,CE_nats_per_tok,delta_vs_greedy"]
for k in names:
    ce = totce[k] / ntok
    lines.append(f"{k},{bpb[k]:.4f},{ce:.4f},{bpb[k]-bpb['greedy']:+.4f}")
open(CSV, "w").write("\n".join(lines) + "\n")
print("[o2] wrote", CSV, flush=True)
for k in names:
    print(f"  {k:12s} BPB={bpb[k]:.4f}", flush=True)
print(f"[o2] SCHEDULABILITY: MinFlow m1 BPB {bpb['minflow_m1']:.4f} vs windowed-greedy {bpb['greedy']:.4f} "
      f"= {bpb['minflow_m1']-bpb['greedy']:+.4f} ({'better scheduling shrinks BPB' if bpb['minflow_m1']<bpb['greedy']-0.005 else 'MinFlow ~= greedy in BPB -> transfer weak'})", flush=True)
