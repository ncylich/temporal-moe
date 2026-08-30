"""Cache-conditional experts (Skliar et al., arXiv:2412.00099) as a serving-time walker, at THEIR setting:
an LRU cache of C experts per layer (paper: 50-75% of the pool), routing biased toward cached experts,
a top-J guarantee, no training. Per MoE layer and decode token:
    z' = z + lambda * delta_avg[layer] * cached          (ranking only)
    sel = top-k(z') with the top-J experts of the ORIGINAL z always included
    gate weights from the original z over sel               (vLLM: non-selected logits -> -inf)
    loads = sel \\ cache  (counted as expert swaps), cache <- LRU update with capacity C
delta_avg[layer] is the running mean of max(z)-min(z) over tokens seen (online, no calibration).
Prefill is observed unbiased and warms the cache (LRU state after a sequential pass = the C most
recently used experts); loads during prefill are not counted, matching our own swap metric.
Env: TEMPORAL_WALKER=cache_bias TEMPORAL_CB_C=<cache size> TEMPORAL_CB_LAMBDA=<0..1> TEMPORAL_CB_J=<int> TEMPORAL_CB_K=<top-k>
"""
import os
import torch

CFG = {"C": int(os.environ.get("TEMPORAL_CB_C", "64")), "lam": float(os.environ.get("TEMPORAL_CB_LAMBDA", "0.5")),
       "J": int(os.environ.get("TEMPORAL_CB_J", "1")), "k": int(os.environ.get("TEMPORAL_CB_K", "8"))}
STATE = {}          # (req_id, layer) -> {"last": [E] float (-inf = not cached), "t": int}
DELTA = {}          # layer -> [sum, n]
COUNT = [0, 0]      # loads on decode rows, decode rows
NEG = float("-inf")


def reset_counts():
    COUNT[0] = COUNT[1] = 0


def prune(live):
    for key in [k for k in STATE if k[0] not in live]:
        del STATE[key]


def _delta(layer, lt):
    s, n = DELTA.get(layer, [0.0, 0])
    s += float((lt.max(dim=-1).values - lt.min(dim=-1).values).sum()); n += lt.shape[0]
    DELTA[layer] = [s, n]
    return s / n


def _lru_prefill(lt, C, k):
    """LRU state after an unbiased sequential pass over the prompt: the C most recently used experts."""
    T, E = lt.shape
    sel = lt.topk(k, dim=-1).indices                                        # [T,k]
    pos = torch.arange(T, device=lt.device, dtype=torch.float32)[:, None].expand(T, k)
    last = torch.full((E,), NEG, device=lt.device)
    last = last.scatter_reduce(0, sel.reshape(-1), pos.reshape(-1), reduce="amax")
    return _cap(last[None], C)[0], T


def _cap(last, C):
    """Keep at most C cached experts per row (largest last-use); tie-break by expert index."""
    E = last.shape[-1]
    key = torch.where(last > NEG, last + torch.arange(E, device=last.device, dtype=last.dtype) * (0.5 / E), torch.full_like(last, NEG))   # integer timestamps: sub-unit tie-break by expert index survives float32
    thr = key.topk(min(C, E), dim=-1).values[:, -1:]
    return torch.where(key >= thr, last, torch.full_like(last, NEG))


def select(lt, last, delta, lam, J, k):
    """Biased top-k with the top-J-of-original guarantee. lt [D,E] float; last [D,E]. Returns the bool mask [D,E]."""
    D, E = lt.shape
    cached = last > NEG
    zb = lt + lam * delta * cached.float()
    sel = torch.zeros_like(cached).scatter_(1, zb.topk(k, dim=-1).indices, True)
    if J > 0:
        topj = torch.zeros_like(cached).scatter_(1, lt.topk(J, dim=-1).indices, True)
        missing = (topj & ~sel).sum(1, keepdim=True)                         # [D,1] how many to swap in
        cand = sel & ~topj                                                   # droppable selections
        order = zb.masked_fill(~cand, float("inf")).argsort(dim=-1)          # weakest candidates first
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(E, device=lt.device).expand(D, E))
        drop = cand & (rank < missing)
        sel = (sel & ~drop) | topj
    return sel


def apply(layer, router_logits, spans, on):
    """vllm_residency.apply for the cache_bias walker: same span protocol as the dict walker."""
    if not on or spans is None:
        return router_logits
    C, lam, J, k = CFG["C"], CFG["lam"], CFG["J"], CFG["k"]
    out = router_logits.clone()
    dec_rows, dec_keys, o = [], [], 0
    with torch.no_grad():
        for sp in spans:
            req_id, n, is_prefill = sp[0], sp[1], sp[2]
            key = (req_id, layer)
            lt = router_logits[o:o + n].float()
            if is_prefill or n > 1 or key not in STATE:
                _delta(layer, lt)
                last, T = _lru_prefill(lt, C, k)                                   # warm (or re-warm after preemption)
                prev = STATE.get(key)
                STATE[key] = {"last": last, "t": (prev["t"] if prev else 0) + T}
            else:
                dec_rows.append(o); dec_keys.append(key)
            o += n
        if dec_rows:
            lt = router_logits[dec_rows].float()
            delta = _delta(layer, lt)
            last = torch.stack([STATE[kk]["last"] for kk in dec_keys])              # [D,E]
            t = torch.tensor([float(STATE[kk]["t"]) for kk in dec_keys], device=lt.device)[:, None]
            sel = select(lt, last, delta, lam, J, k)
            loads = (sel & ~(last > NEG)).sum().item()
            COUNT[0] += int(loads); COUNT[1] += len(dec_rows)
            last = _cap(torch.where(sel, t, last), C)
            for j, kk in enumerate(dec_keys):
                STATE[kk]["last"] = last[j]; STATE[kk]["t"] += 1
            out[dec_rows] = out[dec_rows].masked_fill(~sel, NEG)
    return out


def stats():
    sw, tk = COUNT
    return sw, tk, (sw / tk if tk else 0.0)
