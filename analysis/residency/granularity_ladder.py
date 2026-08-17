#!/usr/bin/env python3
"""Granularity-law ladder cells for the non-resident-program models (training-free).

Per model: free baseline + the iso-fraction ladder from 12.5% resident down to the model's
fundamental floor k/E (R = f*E, R >= k). Damage = BPB(R) - BPB(free) on the byte-identical
audited slice retokenized per tokenizer (build_qwen_slice.py). min_logit rolling residency,
<=1 swap/token, selection masked / gate weights taken per each model's own convention.

    granularity_ladder.py --model lfm25
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import ABLATIONS                                          # noqa: E402
from temporal.temporal_router import compute_resident_mask_accel     # noqa: E402

DATA = "/workspace/olmoe-adapt/data"
MODELS = {
    "lfm25": {"path": "/dev/shm/lfm25-8b-base", "slice": "lfm25",
              "E": 32, "k": 4, "arch": "lfm"},
    "gemma4": {"path": "/gemma4-26b", "slice": "gemma4",
               "E": 128, "k": 8, "arch": "gemma4"},
}
CFG = {"on": False, "R": 8, "free_set": None, "R_map": None}


def fractions(E, k):
    """12.5% down to the floor k/E, halving-ish on the R lattice used by the qwen grids."""
    ladder = [int(round(f * E)) for f in (0.125, 0.09375, 0.0625, 0.046875, 0.03125,
                                          0.0234375, 0.01953125)]
    return sorted({max(R, k) for R in ladder if R >= k} | {k}, reverse=True)


def patch_lfm():
    from transformers.models.lfm2_moe import modeling_lfm2_moe as m
    orig_fwd = m.Lfm2MoeSparseMoeBlock.forward
    orig_route = m.Lfm2MoeSparseMoeBlock.route_tokens_to_experts

    def fwd(self, hidden_states):
        self._resid_shape = hidden_states.shape[:2]
        return orig_fwd(self, hidden_states)

    def route(self, router_logits):
        if not CFG["on"]:
            return orig_route(self, router_logits)
        # Selection signal is the model's own: sigmoid + expert bias. The scan constrains WHICH
        # experts are selectable; gate weights still come from the raw sigmoid at the selected
        # experts (the model's convention, norm_topk_prob=True renormalizes afterwards).
        scores = router_logits.sigmoid()
        sel_signal = scores + self.expert_bias if self.use_expert_bias else scores
        b, s = self._resid_shape
        lg = sel_signal.view(b, s, -1).transpose(0, 1).contiguous()
        if CFG.get("decode_mode"):     # generation: stateful rule across forwards, prefill free
            import decode_state as _DS
            mask = _DS.route(getattr(self, "_layer_idx", id(self)), lg)
            if mask is None:
                mask = torch.ones_like(lg, dtype=torch.bool)
        else:
            with torch.no_grad():
                mask = compute_resident_mask_accel(lg.float(), CFG["R"], evict="min_logit",
                                                   swaps=1)
        ef = CFG.get("enforce_from", 0)
        if ef:         # instruct protocol: prefill positions free, rule enforced from response
            if CFG.get("cold_start"):
                with torch.no_grad():
                    cm = compute_resident_mask_accel(lg[ef:].float(), CFG["R"],
                                                     evict="min_logit", swaps=1)
                mask = torch.ones_like(mask)
                mask[ef:] = cm
            else:
                mask[:ef] = True
        masked = sel_signal.masked_fill(~mask.transpose(0, 1).reshape(sel_signal.shape),
                                        float("-inf"))
        _, selected = torch.topk(masked, k=self.top_k, dim=-1)
        weights = torch.gather(scores, dim=1, index=selected).type_as(router_logits)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        weights = weights * self.routed_scaling_factor
        return selected, weights

    m.Lfm2MoeSparseMoeBlock.forward = fwd
    m.Lfm2MoeSparseMoeBlock.route_tokens_to_experts = route


def patch_gemma4():
    from transformers.models.gemma4 import modeling_gemma4 as m
    import torch.nn as nn

    def fwd(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)                     # [B*S, E]
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1)
        li = getattr(self, "_layer_idx", None)
        fs = CFG.get("free_set")
        freed = not CFG["on"] or (fs is not None and li in fs)
        if not freed:
            R = CFG["R"]
            rm = CFG.get("R_map")
            if rm is not None and li is not None:
                R = rm.get(li, R)                # per-layer residency budget (allocation cells)
            # scan wants [S, B, E] (seq-first, batch columns independent). Eval
            # runs batch 1 ([B*S, E] == [S, E]); training may set CFG["batch"]=B
            # with rows padded to equal S and per-row enforce_from.
            Bn = CFG.get("batch", 1)
            if Bn > 1:
                assert not CFG.get("decode_mode") and not CFG.get("cold_start"), \
                    "batched constraint supports the warm training path only"
                T_, E_ = expert_scores.shape
                S_ = T_ // Bn
                lg = expert_scores.view(Bn, S_, E_).transpose(0, 1).float()
            else:
                lg = expert_scores.unsqueeze(1).float()
            if CFG.get("decode_mode"):  # generation: stateful rule across forwards, prefill free
                import decode_state as _DS
                mask = _DS.route(getattr(self, "_layer_idx", id(self)), lg)
                if mask is None:
                    mask = torch.ones_like(lg, dtype=torch.bool)
            else:
                with torch.no_grad():
                    mask = compute_resident_mask_accel(lg, R, evict="min_logit", swaps=1)
            ef = CFG.get("enforce_from", 0)
            efs = list(ef) if hasattr(ef, "__len__") else [ef] * Bn
            if Bn > 1:
                for b_, e_ in enumerate(efs):
                    if e_:     # prefill positions free, rule from first response token
                        mask[:e_, b_] = True
                probs_for_topk = router_probabilities.masked_fill(
                    ~mask.transpose(0, 1).reshape(T_, E_), 0.0)
            else:
                if efs[0]:
                    mask[:efs[0]] = True
                probs_for_topk = router_probabilities.masked_fill(
                    ~mask.squeeze(1), 0.0)
        else:
            probs_for_topk = router_probabilities
        top_k_weights, top_k_index = torch.topk(probs_for_topk, k=self.config.top_k_experts,
                                                dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index

    m.Gemma4TextRouter.forward = fwd


def tag_gemma4(model):
    """Index router instances in depth order so free_set/R_map layer indices mean what they say."""
    from transformers.models.gemma4 import modeling_gemma4 as m
    n = 0
    for mod in model.modules():
        if isinstance(mod, m.Gemma4TextRouter):
            mod._layer_idx = n
            n += 1
    return n


def patch_qwen35():
    """qwen3.5 mirror of patch_gemma4: mask router logits to -inf before the stock
    softmax->topk->renorm; shared expert untouched (always resident, as served)."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    import torch.nn.functional as F

    def fwd(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight)         # [T, E]
        li = getattr(self, "_layer_idx", None)
        fs = CFG.get("free_set")
        freed = not CFG["on"] or (fs is not None and li in fs)
        logits_for_topk = router_logits
        if not freed:
            R = CFG["R"]
            rm = CFG.get("R_map")
            if rm is not None and li is not None:
                R = rm.get(li, R)
            Bn = CFG.get("batch", 1)
            if Bn > 1:
                assert not CFG.get("decode_mode") and not CFG.get("cold_start"), \
                    "batched constraint supports the warm training path only"
                T_, E_ = router_logits.shape
                S_ = T_ // Bn
                lg = router_logits.view(Bn, S_, E_).transpose(0, 1).float()
            else:
                T_, E_ = router_logits.shape
                lg = router_logits.unsqueeze(1).float()
            with torch.no_grad():
                mask = compute_resident_mask_accel(lg, R, evict="min_logit", swaps=1)
            ef = CFG.get("enforce_from", 0)
            efs = list(ef) if hasattr(ef, "__len__") else [ef] * Bn
            if Bn > 1:
                for b_, e_ in enumerate(efs):
                    if e_:     # prefill positions free, rule from first response token
                        mask[:e_, b_] = True
                logits_for_topk = router_logits.masked_fill(
                    ~mask.transpose(0, 1).reshape(T_, E_), float("-inf"))
            else:
                if efs[0]:
                    mask[:efs[0]] = True
                logits_for_topk = router_logits.masked_fill(
                    ~mask.squeeze(1), float("-inf"))
        router_probs = F.softmax(logits_for_topk, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    m.Qwen3_5MoeTopKRouter.forward = fwd


def tag_qwen35(model):
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    n = 0
    for mod in model.modules():
        if isinstance(mod, m.Qwen3_5MoeTopKRouter):
            mod._layer_idx = n
            n += 1
    return n


R_LAYER = [8, 12, 16, 24]


def perlayer(A, M, model, ids, D):
    """gemma4 per-layer damage curves d_l(R) + fitted greedy allocation vs uniform, base
    checkpoint only (no adapted surface exists for this model). Mirror of the frontier_qwen.py
    --perlayer stage; smoke anchors abort before the long part."""
    import math
    L = tag_gemma4(model)
    ALL = list(range(L))
    print(f"  [perlayer] {L} MoE routers tagged", flush=True)
    out = os.path.join(ABLATIONS, "perlayer_gemma4.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        fh.write('"# gemma4 per-layer damage d_l(R): solo layer l constrained at R, others free, '
                 'base checkpoint; then fitted greedy allocation vs uniform at iso-memory slot '
                 'budgets. Training-free, min_logit <=1 swap/token, model-convention gating. '
                 'free_set: ALL = every layer free, all_but_l = solo cell. damage = bpb - free '
                 'bpb. Producer: analysis/residency/granularity_ladder.py --perlayer"\n')
        w.writerow(["stage", "cell", "R", "free_set", "R_map", "bpb", "damage", "n_seq", "secs"])

    free = {"v": None}

    def cell(stage, name, R=8, free_set=None, fs_tag="", r_map=None):
        CFG.update(on=True, R=R, free_set=set(free_set) if free_set is not None else None,
                   R_map=r_map)
        t0 = time.time()
        v = bpb(model, ids, D)
        w.writerow([stage, name, R, fs_tag,
                    "" if r_map is None else ";".join(f"{k}:{v2}" for k, v2 in
                                                      sorted(r_map.items())),
                    f"{v:.6f}", "" if free["v"] is None else f"{v-free['v']:+.6f}",
                    A.n_seq, f"{time.time()-t0:.1f}"])
        fh.flush()
        print(f"  [{stage}] {name:24} R={R} BPB={v:.6f} ({time.time()-t0:.0f}s)", flush=True)
        return v

    # smoke: free + impose anchors from the ladder rows, R_map parity, solo sanity
    v_free = cell("smoke", "free", free_set=ALL, fs_tag="ALL")
    free["v"] = v_free
    assert abs(v_free - 0.6449) < 0.02, f"free anchor off: {v_free}"
    v_r8 = cell("smoke", "R8_all")
    assert abs(v_r8 - 0.6996) < 0.02, f"impose anchor off: {v_r8}"
    v_map = cell("smoke", "R8_via_uniform_R_map", r_map={i: 8 for i in ALL})
    assert abs(v_map - v_r8) < 1e-5, f"R_map parity broken: {v_map} vs {v_r8}"
    v_solo = cell("smoke", "solo_L00_R8", free_set=[x for x in ALL if x != 0],
                  fs_tag="all_but_0")
    assert v_free - 0.01 < v_solo < v_r8, f"solo cell implausible: {v_solo}"
    print("  [smoke] ALL SMOKES PASS", flush=True)

    d = {(0, 8): v_solo - v_free}
    for l in range(L):
        for R in R_LAYER:
            if (l, R) in d:
                continue
            d[(l, R)] = cell("layers", f"solo_L{l:02d}_R{R}", R=R,
                             free_set=[x for x in ALL if x != l], fs_tag=f"all_but_{l}") - v_free

    fit = {}
    for l in range(L):
        xs = [math.log(R) for R in R_LAYER]
        ys = [math.log(max(d[(l, R)], 1e-6)) for R in R_LAYER]
        n = len(xs)
        b = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / \
            (n * sum(x * x for x in xs) - sum(xs) ** 2)
        a = (sum(ys) - b * sum(xs)) / n
        fit[l] = (math.exp(a), -b)                      # d_l(R) ~ A * R^-g
    pred = lambda l, R: fit[l][0] * R ** (-fit[l][1])                    # noqa: E731
    for budget in (12 * L, 16 * L):
        alloc = {l: 8 for l in range(L)}
        for _ in range(budget - 8 * L):
            best = max(range(L), key=lambda l: pred(l, alloc[l]) - pred(l, alloc[l] + 1))
            alloc[best] += 1
        print(f"  [alloc] budget={budget}: " +
              " ".join(f"L{l}:{alloc[l]}" for l in range(L)), flush=True)
        cell("alloc", f"fitted_B{budget}", r_map=alloc)
        cell("alloc", f"uniform_B{budget}", R=budget // L)
    fh.close()
    print("PERLAYER GEMMA4 COMPLETE", flush=True)


def bpb(model, ids, divisor):
    tot = ntok = 0
    with torch.no_grad():
        for i in range(ids.shape[0]):
            b = ids[i:i + 1].to("cuda").long()
            lg = model(b).logits[:, :-1]
            tg = b[:, 1:]
            for i0 in range(0, lg.shape[1], 512):
                sl = lg[:, i0:i0 + 512].float()
                tot += float(torch.nn.functional.cross_entropy(
                    sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1),
                    reduction="sum"))
                del sl
            ntok += tg.numel()
            del lg
    return (tot / ntok) / divisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--cells", default=None,
                    help="comma R list overriding the default fraction ladder")
    ap.add_argument("--perlayer", action="store_true",
                    help="per-layer d_l(R) curves + fitted allocation (gemma4 only)")
    A = ap.parse_args()
    M = MODELS[A.model]

    from transformers import AutoModelForCausalLM
    kw = {"dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(M["path"], **kw).to("cuda")
    model.eval()
    {"lfm": patch_lfm, "gemma4": patch_gemma4}[M["arch"]]()

    D = json.load(open(f"{DATA}/bpb_slice_meta_{M['slice']}.json"))["divisor_D"]
    ids = torch.load(f"{DATA}/bpb_slice_ids_{M['slice']}.pt", weights_only=False)[: A.n_seq]

    if A.perlayer:
        assert M["arch"] == "gemma4", "--perlayer is wired for the gemma4 patch only"
        perlayer(A, M, model, ids, D)
        return

    out = os.path.join(ABLATIONS, "granularity_ladder.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        fh.write('"# Granularity-law ladder, training-free, min_logit <=1 swap/token, base '
                 'checkpoints, byte-identical audited slice per tokenizer. damage = bpb - free '
                 'bpb of the same model. Producer: analysis/residency/granularity_ladder.py"\n')
        w.writerow(["model", "E", "k", "cell", "R", "frac_pct", "bpb", "damage", "n_seq", "secs"])

    def cell(name, R=None):
        CFG.update(on=R is not None, R=R or 0)
        t0 = time.time()
        v = bpb(model, ids, D)
        frac = "" if R is None else f"{100*R/M['E']:.2f}"
        w.writerow([A.model, M["E"], M["k"], name, R or "", frac, f"{v:.6f}",
                    "" if R is None else f"{v-free:+.6f}", A.n_seq, f"{time.time()-t0:.1f}"])
        fh.flush()
        print(f"  [{A.model}] {name:12} BPB={v:.6f} ({time.time()-t0:.0f}s)", flush=True)
        return v

    free = None
    free = cell("free")
    assert free < 1.5, f"free BPB implausible ({free}) - slice or model wiring is wrong"
    Rs = ([int(x) for x in A.cells.split(",")] if A.cells
          else fractions(M["E"], M["k"]))
    for R in Rs:
        v = cell(f"R{R}", R=R)
        assert v > free - 0.01, f"constrained beat free ({v} vs {free}) - masking not engaged"
    fh.close()
    print(f"LADDER {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
