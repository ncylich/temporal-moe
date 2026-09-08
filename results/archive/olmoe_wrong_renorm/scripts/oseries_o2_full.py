#!/usr/bin/env python3
"""O-series closing check (orch 0111): the UNWINDOWED BPB pair — greedy-scan vs MinFlow m=1 at FULL
T=4096 (single realistic cold-fill, C=32 exact solver, NO parking) — removes the windowing asterisk
from the headline "greedy ~= hindsight-optimal <=1-swap in actual quality". Frozen base @ R=8, audited
slice (D=3.1089). Also the free-top8 anchor. Usage: oseries_o2_full.py <n_packs>."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts"); sys.path.insert(0, "/workspace/FLAME-MoE")
import olmoe_residency as RES
from temporal.temporal_router import compute_resident_mask
from ortools.graph.python import min_cost_flow

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TW = 4096; K = 8; SCALE = 1_000_000
DD = "/workspace/olmoe-adapt/data/oseries"; meta = json.load(open(f"{DD}/meta.json"))
NS, L, T, E, TOPK = meta["n_seq"], meta["L"], meta["seq"], meta["E"], meta["topk"]
D = json.load(open("/workspace/olmoe-adapt/data/bpb_slice_meta.json"))["divisor_D"]
idx = np.memmap(f"{DD}/idx.u8", np.uint8, "r", shape=(NS, L, T, TOPK))
val = np.memmap(f"{DD}/val.f16", np.float16, "r", shape=(NS, L, T, TOPK))
n = min(N, NS)
print(f"[o2full] {n} packs, FULL T={TW} single cold-fill; schedules: free, greedy, MinFlow m=1 (C=32 exact)", flush=True)


def dense(s, l):
    m = np.zeros((T, E)); np.put_along_axis(m, idx[s, l].astype(np.int64), val[s, l].astype(np.float64), 1); return m


def minflow_mask(win, m, Cx=32):                             # exact 8/token, grow C if infeasible
    for C in (Cx, 48, 64):
        cand = np.argsort(-win, 1)[:, :C]; cm = np.take_along_axis(win, cand, 1)
        IN = lambda t, j: t * C + j; OUT = lambda t, j: C * TW + t * C + j
        HIN = lambda t: 2 * C * TW + t; HOUT = lambda t: 2 * C * TW + TW + t
        S = 2 * C * TW + 2 * TW; Ksk = S + 1; jr = np.arange(C); T1 = TW - 1
        tl, hd, cp, co = [], [], [], []
        def a(t, h, c, o): tl.append(np.asarray(t)); hd.append(np.asarray(h)); cp.append(np.asarray(c, np.int64)); co.append(np.asarray(o, np.int64))
        tt = np.repeat(np.arange(TW), C); jj = np.tile(jr, TW)
        a(tt * C + jj, C * TW + tt * C + jj, np.ones(C * TW), -np.rint(cm.reshape(-1) * SCALE))
        a(np.full(C, S), jr, np.ones(C), np.zeros(C))
        eq = cand[:-1, :, None] == cand[1:, None, :]; tp, ja, jb = np.nonzero(eq)
        a(OUT(tp, ja), IN(tp + 1, jb), np.ones(len(tp)), np.zeros(len(tp)))
        te = np.repeat(np.arange(T1), C); je = np.tile(jr, T1)
        a(OUT(te, je), HIN(te + 1), np.ones(C * T1), np.zeros(C * T1))
        ht = np.arange(1, TW); a(HIN(ht), HOUT(ht), np.full(T1, m), np.zeros(T1))
        a(HOUT(te + 1), IN(te + 1, je), np.ones(C * T1), np.zeros(C * T1))
        a(OUT(np.full(C, T1), jr), np.full(C, Ksk), np.ones(C), np.zeros(C))
        g = min_cost_flow.SimpleMinCostFlow(); nr = C * TW
        g.add_arcs_with_capacity_and_unit_cost(np.concatenate(tl), np.concatenate(hd), np.concatenate(cp), np.concatenate(co))
        g.set_node_supply(int(S), K); g.set_node_supply(int(Ksk), -K)
        if g.solve() == g.OPTIMAL:
            fl = np.array([g.flow(i) for i in range(nr)]).reshape(TW, C)
            mk = np.zeros((TW, E), bool)
            for t in range(TW):
                mk[t, cand[t][fl[t] > 0]] = True
            return mk
    return None


model, tok = RES.load_model(); RES.tag_layers(model); RES.enable_residency(R=8)
ids = torch.load("/workspace/olmoe-adapt/data/bpb_slice_ids.pt")[:n].long()


def bpb_forced(x, masks):
    RES.set_forced(masks)
    with torch.no_grad():
        out = model(x).logits.float()
        ce = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
    return ce.item()


tot = {"free_top8": 0.0, "greedy": 0.0, "minflow_m1": 0.0}; ntok = 0
for si in range(n):
    x = ids[si:si + 1].to("cuda")
    RES.disable_residency(); tot["free_top8"] += bpb_forced(x, None); RES.enable_residency(R=8)
    gmasks, mmasks = [], []
    for l in range(L):
        mass = dense(si, l)
        lg = torch.from_numpy(np.where(mass > 0, np.log(mass + 1e-12), -1e9)).float().unsqueeze(1)
        gmasks.append(compute_resident_mask(lg, K, evict="min_logit").squeeze(1))       # [T,1,E]
        mk = minflow_mask(mass, 1); mmasks.append(torch.from_numpy(mk).unsqueeze(1))
    tot["greedy"] += bpb_forced(x, gmasks)
    tot["minflow_m1"] += bpb_forced(x, mmasks)
    ntok += x[:, 1:].numel()
    print(f"[o2full] pack {si+1}/{n} done", flush=True)
RES.set_forced(None)

bpb = {k: tot[k] / ntok / D for k in tot}
CSV = "/workspace/FLAME-MoE/results/ablations/olmoe_minflow_bpb_full.csv"
open(CSV, "w").write(
    "# O-series closing: UNWINDOWED (full-4096, single cold-fill) BPB pair, C=32 exact solver, %d packs, D=%.4f.\n" % (n, D) +
    "# Removes the windowing asterisk from the headline: greedy vs hindsight-optimal m=1 in ACTUAL quality.\n" +
    "schedule,BPB,delta_vs_greedy\n" +
    "\n".join(f"{k},{bpb[k]:.4f},{bpb[k]-bpb['greedy']:+.4f}" for k in ["free_top8", "greedy", "minflow_m1"]) + "\n")
print("[o2full] wrote", CSV, flush=True)
for k in ["free_top8", "greedy", "minflow_m1"]:
    print(f"  {k:12s} BPB={bpb[k]:.4f}", flush=True)
print(f"[o2full] UNWINDOWED HEADLINE: greedy {bpb['greedy']:.4f} vs MinFlow m1 {bpb['minflow_m1']:.4f} "
      f"= {bpb['minflow_m1']-bpb['greedy']:+.4f} BPB (greedy ~= hindsight-optimal <=1-swap)", flush=True)
