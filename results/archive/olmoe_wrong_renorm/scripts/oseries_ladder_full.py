#!/usr/bin/env python3
"""COMPLETE O-0/O-1 captured-mass ladder — FULL 4096-token sequences (single realistic cold-fill, NO
windowing) via per-token top-C candidate pruning (a schedule never holds a rank>C expert, so this is
near-exact; C=16 with 8 slots leaves ample slack). Graph built fully vectorized (numpy). Policies:
  static-best-8 <= greedy-scan <= MinFlow(m) <= per-token top-8 bound.
Usage: oseries_ladder_full.py <m_csv> <n_seq> <out_csv> [C]
"""
import os, sys, json, numpy as np, torch, time
sys.path.insert(0, "/workspace/olmoe-adapt/scripts"); sys.path.insert(0, "/workspace/FLAME-MoE")
from temporal.temporal_router import compute_resident_mask
from ortools.graph.python import min_cost_flow

MS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1]
N_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 32
OUT_CSV = sys.argv[3] if len(sys.argv) > 3 else "/workspace/FLAME-MoE/results/ablations/olmoe_minflow_full.csv"
C = int(sys.argv[4]) if len(sys.argv) > 4 else 16
D = "/workspace/olmoe-adapt/data/oseries"; meta = json.load(open(f"{D}/meta.json"))
NS, L, T, E, TOPK, K = meta["n_seq"], meta["L"], meta["seq"], meta["E"], meta["topk"], 8
SCALE = 1_000_000
idx = np.memmap(f"{D}/idx.u8", dtype=np.uint8, mode="r", shape=(NS, L, T, TOPK))
val = np.memmap(f"{D}/val.f16", dtype=np.float16, mode="r", shape=(NS, L, T, TOPK))
n_seq = min(N_REQ, NS)
print(f"[full] {NS} stored, using {n_seq} seqs FULL T={T}; prune top-C={C}; m={MS}", flush=True)


def dense_mass(s, l):
    m = np.zeros((T, E), dtype=np.float64)
    np.put_along_axis(m, idx[s, l].astype(np.int64), val[s, l].astype(np.float64), axis=1)
    return m


def minflow_full(mass, m):
    # top-C candidate lane + a PARKING lane (0-reward hold of a slot on a now-low expert), which keeps
    # the graph small (C=16) yet always feasible: any resident whose expert leaves the top-C parks its
    # slot with no swap (a low-mass hold contributes ~0, so 0 reward is a tight approximation).
    cand = np.argsort(-mass, axis=1)[:, :C]
    cmass = np.take_along_axis(mass, cand, axis=1)
    IN = lambda t, j: t * C + j
    OUT = lambda t, j: C * T + t * C + j
    HIN = lambda t: 2 * C * T + t
    HOUT = lambda t: 2 * C * T + T + t
    PIN = lambda t: 2 * C * T + 2 * T + t
    POUT = lambda t: 2 * C * T + 3 * T + t
    S = 2 * C * T + 4 * T; Ksink = S + 1
    jrow = np.arange(C); T1 = T - 1
    tails, heads, caps, costs = [], [], [], []

    def add(tl, hd, cp, co):
        tails.append(np.asarray(tl)); heads.append(np.asarray(hd))
        caps.append(np.asarray(cp, np.int64)); costs.append(np.asarray(co, np.int64))
    tt = np.repeat(np.arange(T), C); jj = np.tile(jrow, T)
    add(tt * C + jj, C * T + tt * C + jj, np.ones(C * T), -np.rint(cmass.reshape(-1) * SCALE))  # reward
    add(np.full(C, S), jrow, np.ones(C), np.zeros(C))                                            # cold -> candidates
    add([S], [PIN(0)], [K], [0])                                                                 # cold -> parking (spare slots)
    eq = cand[:-1, :, None] == cand[1:, None, :]; tp, ja, jb = np.nonzero(eq)
    add(OUT(tp, ja), IN(tp + 1, jb), np.ones(len(tp)), np.zeros(len(tp)))                        # stay cand->cand
    te = np.repeat(np.arange(T1), C); je = np.tile(jrow, T1)
    add(OUT(te, je), HIN(te + 1), np.ones(C * T1), np.zeros(C * T1))                             # evict cand->hub
    add(OUT(te, je), PIN(te + 1), np.ones(C * T1), np.zeros(C * T1))                             # cand drops -> park (no swap)
    ht = np.arange(1, T)
    add(HIN(ht), HOUT(ht), np.full(T1, m), np.zeros(T1))                                         # hub cap m (=swaps)
    add(HOUT(te + 1), IN(te + 1, je), np.ones(C * T1), np.zeros(C * T1))                         # admit hub->cand
    add(PIN(np.arange(T)), POUT(np.arange(T)), np.full(T, K), np.zeros(T))                       # park hold (cap 8)
    add(POUT(np.arange(T1)), PIN(np.arange(1, T)), np.full(T1, K), np.zeros(T1))                 # park stays park (no swap)
    add(POUT(np.arange(T1)), HIN(np.arange(1, T)), np.full(T1, K), np.zeros(T1))                 # park -> hub (swap out)
    add(OUT(np.full(C, T1), jrow), np.full(C, Ksink), np.ones(C), np.zeros(C))                   # drain candidates
    add([POUT(T1)], [Ksink], [K], [0])                                                           # drain parking
    smcf = min_cost_flow.SimpleMinCostFlow()
    smcf.add_arcs_with_capacity_and_unit_cost(np.concatenate(tails), np.concatenate(heads),
                                              np.concatenate(caps), np.concatenate(costs))
    smcf.set_node_supply(int(S), K); smcf.set_node_supply(int(Ksink), -K)
    return (-smcf.optimal_cost() / SCALE) if smcf.solve() == smcf.OPTIMAL else float("nan")


acc = {m: np.zeros(L) for m in MS}; g_acc = np.zeros(L); s_acc = np.zeros(L); b_acc = np.zeros(L); tot = 0
t0 = time.time()
for s in range(n_seq):
    for l in range(L):
        mass = dense_mass(s, l)
        srt = -np.sort(-mass, axis=1); b_acc[l] += srt[:, :K].sum()
        s_acc[l] += mass[:, np.argsort(-mass.sum(0))[:K]].sum()
        lg = torch.from_numpy(np.where(mass > 0, np.log(mass + 1e-12), -1e9)).float().unsqueeze(1)
        gm = compute_resident_mask(lg, K, evict="min_logit").squeeze(1).numpy(); g_acc[l] += (gm * mass).sum()
        for m in MS:
            acc[m][l] += minflow_full(mass, m)
    tot += T
    print(f"[full] seq {s+1}/{n_seq} done ({(time.time()-t0)/(s+1):.1f}s/seq)", flush=True)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
hdr = ["# COMPLETE captured-mass ladder — FULL 4096-token seqs (single cold-fill), per-token top-C candidate",
       f"# pruning (C={C}, near-exact). {n_seq} audited-slice seqs. Reward=base softmax mass (never renorm).",
       f"# mean served mass/token per layer. m in {MS}. captured in [0,1]/token.",
       "layer,static_best8,greedy_scan," + ",".join(f"minflow_m{m}" for m in MS) + ",top8_bound"]
for l in range(L):
    row = [f"{s_acc[l]/tot:.5f}", f"{g_acc[l]/tot:.5f}"] + [f"{acc[m][l]/tot:.5f}" for m in MS] + [f"{b_acc[l]/tot:.5f}"]
    hdr.append(f"{l}," + ",".join(row))
agg = [s_acc.sum()/tot/L, g_acc.sum()/tot/L] + [acc[m].sum()/tot/L for m in MS] + [b_acc.sum()/tot/L]
hdr.append("ALL," + ",".join(f"{v:.5f}" for v in agg))
open(OUT_CSV, "w").write("\n".join(hdr) + "\n")
gg, mf = g_acc.sum()/tot/L, acc[MS[0]].sum()/tot/L
print(f"[full] wrote {OUT_CSV}", flush=True)
print(f"[full] AGG static={agg[0]:.5f} greedy={gg:.5f} " + " ".join(f"m{m}={acc[m].sum()/tot/L:.5f}" for m in MS) + f" bound={agg[-1]:.5f}", flush=True)
print(f"[full] KILL-CHECK m={MS[0]}: minflow-greedy=+{mf-gg:.5f} ({'~=greedy STOP' if mf-gg<0.002 else 'MinFlow>greedy proceed'})", flush=True)
