#!/usr/bin/env python3
"""O-series close, larger-N PARALLEL (orch 0128): unwindowed paired greedy-vs-MinFlow-m1 BPB, full-4096,
C=32 exact. The 384 (pack x layer) MinFlow solves are independent -> multiprocessing Pool (each OR-Tools
solve is single-threaded) instead of a serial loop. Then GPU forwards for per-pack BPB -> paired mean
diff +/- paired-SE. Appends to olmoe_minflow_bpb.csv. Usage: oseries_paired_mp.py <n_packs> <workers>"""
import sys, json, numpy as np
from multiprocessing import Pool
from ortools.graph.python import min_cost_flow

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
TW = 4096; K = 8; SCALE = 1_000_000
dd = "/workspace/olmoe-adapt/data/oseries"; meta = json.load(open(dd + "/meta.json"))
NS, L, T, E, TK = meta["n_seq"], meta["L"], meta["seq"], meta["E"], meta["topk"]
D = json.load(open("/workspace/olmoe-adapt/data/bpb_slice_meta.json"))["divisor_D"]
_idx = np.memmap(dd + "/idx.u8", np.uint8, "r", shape=(NS, L, T, TK))
_val = np.memmap(dd + "/val.f16", np.float16, "r", shape=(NS, L, T, TK))


def _dense(s, l):
    a = np.zeros((T, E)); np.put_along_axis(a, _idx[s, l].astype(np.int64), _val[s, l].astype(np.float64), 1); return a


def _minflow(win, m):
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


def worker(sl):
    s, l = sl
    return sl, np.packbits(_minflow(_dense(s, l), 1))          # packed to shrink IPC


if __name__ == "__main__":
    import os, time
    tasks = [(s, l) for s in range(N) for l in range(L)]
    MASKS = f"{dd}/paired_masks_N{N}.npz"                       # checkpoint so a GPU-phase OOM can't waste the solves
    if os.path.exists(MASKS):
        z = np.load(MASKS); res = {tuple(int(x) for x in k.split("_")): z[k] for k in z.files}
        print(f"[mp] loaded {len(res)} cached masks from {MASKS}; skipping solves.", flush=True)
    else:
        t0 = time.time()
        print(f"[mp] {len(tasks)} MinFlow solves over {WORKERS} workers ...", flush=True)
        with Pool(WORKERS) as p:
            res = {}
            for i, (sl, packed) in enumerate(p.imap_unordered(worker, tasks, chunksize=1)):
                res[sl] = packed
                if (i + 1) % 32 == 0:
                    print(f"[mp] {i+1}/{len(tasks)} solves ({(time.time()-t0):.0f}s)", flush=True)
        np.savez(MASKS, **{f"{s}_{l}": res[(s, l)] for (s, l) in res})
        print(f"[mp] all solves done in {(time.time()-t0)/60:.1f} min, cached -> {MASKS}. GPU forwards next.", flush=True)

    import torch
    sys.path.insert(0, "/workspace/olmoe-adapt/scripts"); sys.path.insert(0, "/workspace/FLAME-MoE")
    import olmoe_residency as RES
    from temporal.temporal_router import compute_resident_mask
    model, tok = RES.load_model(); RES.tag_layers(model); RES.enable_residency(R=8)
    ids = torch.load("/workspace/olmoe-adapt/data/bpb_slice_ids.pt")[:N].long()

    def bpb_one(x, masks):
        RES.set_forced(masks)
        with torch.no_grad():
            o = model(x).logits.float()
            ce = torch.nn.functional.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)), x[:, 1:].reshape(-1), reduction="sum").item()
        return ce / x[:, 1:].numel() / D

    gr, mf = [], []
    for s in range(N):
        x = ids[s:s + 1].to("cuda"); gm, mm = [], []
        for l in range(L):
            lgm = np.unpackbits(res[(s, l)])[:TW * E].reshape(TW, E).astype(bool)
            mm.append(torch.from_numpy(lgm).unsqueeze(1))
            mass = _dense(s, l); lg = torch.from_numpy(np.where(mass > 0, np.log(mass + 1e-12), -1e9)).float().unsqueeze(1)
            gm.append(compute_resident_mask(lg, K, evict="min_logit"))
        gr.append(bpb_one(x, gm)); mf.append(bpb_one(x, mm))
        print(f"[paired] pack {s+1}/{N}: greedy={gr[-1]:.4f} minflow={mf[-1]:.4f} diff={gr[-1]-mf[-1]:+.4f}", flush=True)
    RES.set_forced(None)
    gr = np.array(gr); mf = np.array(mf); diff = gr - mf
    md, se = diff.mean(), diff.std(ddof=1) / np.sqrt(N)
    print(f"[paired] N={N} greedy={gr.mean():.4f} MinFlow_m1={mf.mean():.4f}", flush=True)
    print(f"[paired] PAIRED diff (greedy - MinFlow) = {md:.4f} +/- {se:.4f} (paired SE); t={md/se:.1f}", flush=True)
    with open("/workspace/FLAME-MoE/results/ablations/olmoe_minflow_bpb.csv", "a") as f:
        f.write(f"# UNWINDOWED PAIRED (orch 0128) N={N} full-4096 C=32: greedy {gr.mean():.4f} MinFlow_m1 {mf.mean():.4f}; "
                f"paired diff {md:.4f} +/- {se:.4f} BPB (t={md/se:.1f}) -> MinFlow beats greedy by {md:.4f} at full sequence.\n")
    print("[paired] appended.", flush=True)
