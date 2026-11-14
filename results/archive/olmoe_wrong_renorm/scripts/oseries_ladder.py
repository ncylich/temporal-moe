#!/usr/bin/env python3
"""O-0/O-1 captured-mass ladder (orch 0090/0097). Replays 4 scheduling policies over the stored top-24
softmax-mass fields (zero model evals), captured mass per layer:
  static-best-8  <=  greedy-scan  <=  MinFlow(m)  <=  per-token top-8 bound.
MinFlow = EXACT min-cost flow: 8 slot-units through experts x time, node cap 1 per (e,t), stay arcs
collect base-softmax reward (never renormalized), switch arcs through a per-token admission hub of
capacity m, cold fill = 8 free admissions at window start. For solver tractability the 4096 sequence is
processed in TW-token windows (cold-fill per window); ALL policies use the same windows, so the
MinFlow-vs-greedy comparison (the O-0 kill rule) is apples-to-apples.

Usage: oseries_ladder.py <m_csv e.g. 1 or 1,2,4> <n_seq> <out_csv> [tw]
Kill rule (O-0): if MinFlow(1) ~= greedy in captured mass, report and stop the O-series.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts"); sys.path.insert(0, "/workspace/FLAME-MoE")
from temporal.temporal_router import compute_resident_mask
from ortools.graph.python import min_cost_flow

MS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1]
N_SEQ_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT_CSV = sys.argv[3] if len(sys.argv) > 3 else "/workspace/FLAME-MoE/results/ablations/olmoe_minflow_calib.csv"
TW = int(sys.argv[4]) if len(sys.argv) > 4 else 512
D = "/workspace/olmoe-adapt/data/oseries"; meta = json.load(open(f"{D}/meta.json"))
NS, L, T, E, TOPK, K = meta["n_seq"], meta["L"], meta["seq"], meta["E"], meta["topk"], 8
SCALE = 1_000_000
idx = np.memmap(f"{D}/idx.u8", dtype=np.uint8, mode="r", shape=(NS, L, T, TOPK))
val = np.memmap(f"{D}/val.f16", dtype=np.float16, mode="r", shape=(NS, L, T, TOPK))
n_seq = min(N_SEQ_REQ, NS); NW = T // TW
print(f"[ladder] {NS} stored, using {n_seq} seqs x {NW} windows of {TW}; L={L} E={E} m={MS}", flush=True)


def dense_mass(s, l):
    m = np.zeros((T, E), dtype=np.float64)
    np.put_along_axis(m, idx[s, l].astype(np.int64), val[s, l].astype(np.float64), axis=1)
    return m


def build_topology(m):
    e = np.arange(E); tails, heads, caps = [], [], []
    tt = np.repeat(np.arange(TW), E); ee = np.tile(e, TW)
    tails.append(tt * E + ee); heads.append(E * TW + tt * E + ee); caps.append(np.ones(E * TW, np.int64))  # reward
    tails.append(np.full(E, 2 * E * TW + 2 * TW)); heads.append(np.arange(E)); caps.append(np.ones(E, np.int64))  # cold
    for t in range(TW - 1):
        bo = E * TW + t * E; bi = (t + 1) * E
        tails.append(bo + e); heads.append(bi + e); caps.append(np.ones(E, np.int64))                       # stay
        tails.append(bo + e); heads.append(np.full(E, 2 * E * TW + t + 1)); caps.append(np.ones(E, np.int64))  # evict
        tails.append(np.array([2 * E * TW + t + 1])); heads.append(np.array([2 * E * TW + TW + t + 1])); caps.append(np.array([m], np.int64))  # hub cap m
        tails.append(np.full(E, 2 * E * TW + TW + t + 1)); heads.append(bi + e); caps.append(np.ones(E, np.int64))  # admit
    tails.append(E * TW + (TW - 1) * E + e); heads.append(np.full(E, 2 * E * TW + 2 * TW + 1)); caps.append(np.ones(E, np.int64))  # drain
    return (np.concatenate(tails), np.concatenate(heads), np.concatenate(caps), E * TW,
            2 * E * TW + 2 * TW, 2 * E * TW + 2 * TW + 1)


TOPO = {m: build_topology(m) for m in MS}


def minflow_captured(win, m):                            # win: [TW, E] masses
    tails, heads, caps, nr, S, Ksink = TOPO[m]
    costs = np.zeros(len(tails), dtype=np.int64)
    costs[:nr] = -np.rint(win.reshape(-1) * SCALE).astype(np.int64)
    smcf = min_cost_flow.SimpleMinCostFlow()
    smcf.add_arcs_with_capacity_and_unit_cost(tails, heads, caps, costs)
    smcf.set_node_supply(int(S), K); smcf.set_node_supply(int(Ksink), -K)
    return (-smcf.optimal_cost() / SCALE) if smcf.solve() == smcf.OPTIMAL else float("nan")


acc = {m: np.zeros(L) for m in MS}; g_acc = np.zeros(L); s_acc = np.zeros(L); b_acc = np.zeros(L); tot_tok = 0
for s in range(n_seq):
    for l in range(L):
        mass = dense_mass(s, l)
        for w in range(NW):
            win = mass[w * TW:(w + 1) * TW]              # [TW,E]
            srt = -np.sort(-win, axis=1)
            b_acc[l] += srt[:, :K].sum()                 # top-8 bound
            tot = win.sum(0); s_acc[l] += win[:, np.argsort(-tot)[:K]].sum()   # static-best-8
            lg = torch.from_numpy(np.where(win > 0, np.log(win + 1e-12), -1e9)).float().unsqueeze(1)
            gm = compute_resident_mask(lg, K, evict="min_logit").squeeze(1).numpy()
            g_acc[l] += (gm * win).sum()                 # greedy scan
            for m in MS:
                acc[m][l] += minflow_captured(win, m)    # exact MinFlow
    tot_tok += T
    print(f"[ladder] seq {s+1}/{n_seq} done", flush=True)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
hdr = ["# O-0/O-1 captured-mass ladder (mean served base-softmax mass per token, per layer; reward never renormalized).",
       f"# {n_seq} audited-slice seqs, windowed at TW={TW} (cold-fill per window) for exact-MinFlow tractability;",
       f"# all policies share the windows so MinFlow-vs-greedy is apples-to-apples. m in {MS}. captured in [0,1]/token.",
       f"# NOTE base router softmax is FLAT: top-8 covers ~0.406, top-24 ~0.684 (plan's >=0.995 premise fails); top-24 still suffices (schedules serve top-8ish).",
       "layer,static_best8,greedy_scan," + ",".join(f"minflow_m{m}" for m in MS) + ",top8_bound"]
tt = tot_tok
for l in range(L):
    row = [f"{s_acc[l]/tt:.5f}", f"{g_acc[l]/tt:.5f}"] + [f"{acc[m][l]/tt:.5f}" for m in MS] + [f"{b_acc[l]/tt:.5f}"]
    hdr.append(f"{l}," + ",".join(row))
agg = [s_acc.sum()/tt/L, g_acc.sum()/tt/L] + [acc[m].sum()/tt/L for m in MS] + [b_acc.sum()/tt/L]
hdr.append("ALL," + ",".join(f"{v:.5f}" for v in agg))
open(OUT_CSV, "w").write("\n".join(hdr) + "\n")
gg, mf = g_acc.sum()/tt/L, acc[MS[0]].sum()/tt/L
print(f"[ladder] wrote {OUT_CSV}", flush=True)
print(f"[ladder] AGG static={agg[0]:.5f} greedy={gg:.5f} minflow_m{MS[0]}={mf:.5f} top8bound={agg[-1]:.5f}", flush=True)
print(f"[ladder] KILL-CHECK m={MS[0]}: minflow-greedy gain={mf-gg:.5f} "
      f"({'~= greedy -> STOP O-series' if mf-gg < 0.002 else 'MinFlow > greedy -> proceed'})", flush=True)
