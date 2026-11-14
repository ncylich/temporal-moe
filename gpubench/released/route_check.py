#!/usr/bin/env python3
"""Routing-equality gate for the released-model routing modes of the CUDA engine (patch 0006).

The engine (LLAMA_TEMPORAL_ROUTE=decode|skliar, LLAMA_TEMPORAL_ROUTE_DUMP2=<file>) writes, per MoE layer and
token, the router logits it routed on and the expert ids it selected. This script feeds the same logits to
the Python references, imported unmodified, and checks that every selected set is equal:

    decode : analysis/residency/decode_state.py   prefill() over the prompt, step() per generated token
    skliar : analysis/residency/cache_bias.py      apply() with the vLLM span protocol
    both   : prompt tokens must be the stock top-k (routed freely)

Because both sides see the same logits, the check isolates the rule implementation from the forward pass
(quantization, kernels). One dump may hold several sequences; a new sequence starts at every prompt batch.

    python3 gpubench/released/route_check.py --dump route.bin --mode skliar --K 8 --E 256 \
        --C 8 --lam 0.4 [--out check.json]
"""
import argparse, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "analysis", "residency"))


def read_dump(path, K, E):
    rec = 4 + K + E
    raw = np.fromfile(path, dtype=np.int32)
    assert raw.size % rec == 0, f"dump size {raw.size} not a multiple of {rec}"
    raw = raw.reshape(-1, rec)
    hdr = raw[:, :4]; ids = raw[:, 4:4 + K]; lg = raw[:, 4 + K:].copy().view(np.float32)
    return hdr, ids, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--mode", choices=["decode", "skliar"], required=True)
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--C", type=int, default=8); ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--J", type=int, default=1); ap.add_argument("--out")
    a = ap.parse_args()
    os.environ.update(TEMPORAL_CB_C=str(a.C), TEMPORAL_CB_LAMBDA=str(a.lam), TEMPORAL_CB_J=str(a.J),
                      TEMPORAL_CB_K=str(a.K), TEMPORAL_DECODE="eager")
    import torch
    import decode_state as DS
    import cache_bias as CB
    DS.DEC.update(on=True, R=a.K, swaps=1)

    hdr, ids, lg = read_dump(a.dump, a.K, a.E)
    n = len(hdr)
    # group consecutive records of one op: (layer, n_tokens) runs of n_tokens records
    # a record with layer -1 is the engine's sequence reset (position 0); the dump starts inside a sequence
    i, req = 0, 1
    stats = {"prompt_tokens": 0, "prompt_equal": 0, "decode_tokens": 0, "decode_equal": 0, "loads_equal": 0,
             "first_mismatch": None, "sequences": 1, "layers": set(), "prompt_chunks_continued": 0}
    last_kind = {}
    DS.reset()
    while i < n:
        L, nt = int(hdr[i, 0]), int(hdr[i, 1])
        if L == -1:                                        # new sequence
            DS.reset(); req += 1; last_kind = {}; stats["sequences"] += 1
            i += 1; continue
        stats["layers"].add(L)
        blk = slice(i, i + nt)
        x = torch.from_numpy(lg[blk])                      # [nt, E]
        eng = ids[blk]
        if nt > 1:                                         # a prompt batch
            cont = L in last_kind                          # an earlier batch of this sequence: carry the state
            stats["prompt_chunks_continued"] += cont
            top = x.topk(a.K, dim=-1).indices.numpy()
            for t in range(nt):
                stats["prompt_tokens"] += 1
                ok = set(top[t].tolist()) == set(eng[t].tolist())
                stats["prompt_equal"] += ok
                if not ok and stats["first_mismatch"] is None:
                    stats["first_mismatch"] = {"record": i + t, "layer": L, "kind": "prompt"}
            if a.mode == "decode":                         # observe-only scan seeds the state; a continued
                (DS.observe_chunk if cont else DS.prefill)(L, x.unsqueeze(1))   # chunk carries it (observe_chunk)
            else:
                CB.apply(L, x, [(req, nt, True)], True)    # unbiased warm, not counted
        else:
            if a.mode == "decode":
                res = DS.step(L, x)[0]                     # [E] bool, resident = selected (R = K)
                ref = set(torch.nonzero(res).flatten().tolist())
                ref_loads = None
            else:
                before = CB.COUNT[0]
                out = CB.apply(L, x, [(req, 1, False)], True)[0]
                if torch.isfinite(out).all():              # first token of a sequence without a prompt:
                    ref = set(x[0].topk(a.K).indices.tolist())   # the reference warms the cache and routes freely
                    ref_loads = -1
                else:
                    ref = set(torch.nonzero(torch.isfinite(out)).flatten().tolist())
                    ref_loads = CB.COUNT[0] - before
            got = set(eng[0].tolist())
            stats["decode_tokens"] += 1
            ok = ref == got
            stats["decode_equal"] += ok
            eng_loads = int(hdr[i, 3])
            if ref_loads is not None:
                stats["loads_equal"] += (ref_loads == eng_loads)
            if not ok and stats["first_mismatch"] is None:
                stats["first_mismatch"] = {"record": i, "layer": L, "kind": "decode", "ref": sorted(ref), "engine": sorted(got)}
        last_kind[L] = "prompt" if nt > 1 else "decode"
        i += nt
    stats["layers"] = len(stats["layers"])
    print(json.dumps(stats))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({**vars(a), **stats}, f); f.write("\n")


if __name__ == "__main__":
    main()
