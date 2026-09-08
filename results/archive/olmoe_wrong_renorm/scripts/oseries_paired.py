#!/usr/bin/env python3
"""O-series close, larger-N (orch 0128): unwindowed PAIRED greedy-vs-MinFlow-m1 BPB, full-4096 single
cold-fill, C=32 exact solver. Records PER-PACK BPB for greedy and MinFlow, reports the paired mean
difference +/- paired-SE (std of per-pack diffs / sqrt(N)) so the ~0.035 magnitude has an error bar.
Appends an unwindowed/N row to olmoe_minflow_bpb.csv. Fixes the greedy-mask batch-dim shape.
Usage: oseries_paired.py <n_packs>"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts"); sys.path.insert(0, "/workspace/FLAME-MoE")
import olmoe_residency as RES
from temporal.temporal_router import compute_resident_mask
from ortools.graph.python import min_cost_flow

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
TW = 4096; K = 8; SCALE = 1_000_000
D = json.load(open("/workspace/olmoe-adapt/data/bpb_slice_meta.json"))["divisor_D"]
dd = "/workspace/olmoe-adapt/data/oseries"; meta = json.load(open(dd + "/meta.json"))
NS, L, T, E, TK = meta["n_seq"], meta["L"], meta["seq"], meta["E"], meta["topk"]
idx = np.memmap(dd + "/idx.u8", np.uint8, "r", shape=(NS, L, T, TK))
val = np.memmap(dd + "/val.f16", np.float16, "r", shape=(NS, L, T, TK))
n = min(N, NS)


def dense(s, l):
    a = np.zeros((T, E)); np.put_along_axis(a, idx[s, l].astype(np.int64), val[s, l].astype(np.float64), 1); return a


def minflow_mask(win, m):                                     # exact 8/token, grow C until feasible
    for C in (32, 48, 64):
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


def bpb_one(x, masks):
    RES.set_forced(masks)
    with torch.no_grad():
        o = model(x).logits.float()
        ce = torch.nn.functional.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)), x[:, 1:].reshape(-1), reduction="sum").item()
    return ce / x[:, 1:].numel() / D


gr_bpb, mf_bpb = [], []
for s in range(n):
    x = ids[s:s + 1].to("cuda")
    gm, mm = [], []
    for l in range(L):
        mass = dense(s, l)
        lg = torch.from_numpy(np.where(mass > 0, np.log(mass + 1e-12), -1e9)).float().unsqueeze(1)
        gm.append(compute_resident_mask(lg, K, evict="min_logit"))          # [T,1,E] correct shape
        mm.append(torch.from_numpy(minflow_mask(mass, 1)).unsqueeze(1))     # [T,1,E]
    gr_bpb.append(bpb_one(x, gm)); mf_bpb.append(bpb_one(x, mm))
    print(f"[paired] pack {s+1}/{n}: greedy={gr_bpb[-1]:.4f} minflow={mf_bpb[-1]:.4f} diff={gr_bpb[-1]-mf_bpb[-1]:+.4f}", flush=True)
RES.set_forced(None)

gr = np.array(gr_bpb); mf = np.array(mf_bpb); diff = gr - mf
mean_d = diff.mean(); se_d = diff.std(ddof=1) / np.sqrt(n)
print(f"[paired] N={n}  greedy={gr.mean():.4f}  MinFlow_m1={mf.mean():.4f}", flush=True)
print(f"[paired] PAIRED diff (greedy - MinFlow) = {mean_d:.4f} +/- {se_d:.4f} (paired SE); t={mean_d/se_d:.1f}", flush=True)
CSV = "/workspace/FLAME-MoE/results/ablations/olmoe_minflow_bpb.csv"
with open(CSV, "a") as f:
    f.write(f"# UNWINDOWED PAIRED (orch 0128) N={n} packs, full-4096, C=32 exact: greedy {gr.mean():.4f}, MinFlow_m1 {mf.mean():.4f}, "
            f"paired diff {mean_d:.4f} +/- {se_d:.4f} BPB (t={mean_d/se_d:.1f}); MinFlow beats greedy by {mean_d:.4f} at full sequence.\n")
print("[paired] appended to", CSV, flush=True)
