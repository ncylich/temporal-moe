#!/usr/bin/env python3
"""Per-layer Wasserstein displacement of router probabilities under residency.

For a base IT checkpoint (no training), teacher-forces WildChat prompts and
measures, per MoE layer, the 1-Wasserstein distance with unit ground metric
(= total variation, the standard choice for categorical supports) between the
free router distribution and the residency-constrained one, two ways:

  w1_imposed    mask from the scan applied to the FREE forward's own logits,
                renormalised: the instantaneous per-layer displacement on
                identical inputs (no compounding).
  w1_endtoend   a second, fully constrained forward (CFG on, R=k, from
                position 0): includes drift compounded through depth.

LFM's router is sigmoid-scored; its distributions are sum-normalised for
comparability and flagged in the output. Constraint: R=k, min-logit, 1 swap
per token, applied from position 0 (imposition convention, as in the locus
measurements).

    router_wasserstein.py --family qwen35 --model-path .../qwen35-... --R 8
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import granularity_ladder as GL                                      # noqa: E402
from paths import ABLATIONS                                          # noqa: E402

PROMPTS = "/workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl"


def routers_of(model, family):
    if family == "gemma4":
        from transformers.models.gemma4 import modeling_gemma4 as m
        return [mod for mod in model.modules() if isinstance(mod, m.Gemma4TextRouter)]
    if family == "qwen35":
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
        return [mod for mod in model.modules() if isinstance(mod, m.Qwen3_5MoeTopKRouter)]
    if family == "lfm":
        from transformers.models.lfm2_moe import modeling_lfm2_moe as m
        return [mod for mod in model.modules() if isinstance(mod, m.Lfm2MoeSparseMoeBlock)]
    raise ValueError(family)


def hook_capture(routers, family, store):
    handles = []
    for li, r in enumerate(routers):
        if family == "gemma4":                   # logits from the proj Linear
            h = r.proj.register_forward_hook(
                lambda _m, _i, out, li=li: store.setdefault(li, []).append(
                    out.detach().float()))
        elif family == "qwen35":                 # router returns (logits, w, idx)
            h = r.register_forward_hook(
                lambda _m, _i, out, li=li: store.setdefault(li, []).append(
                    out[0].detach().float()))
        else:                                    # lfm: gate Linear inside the block
            h = r.gate.register_forward_hook(
                lambda _m, _i, out, li=li: store.setdefault(li, []).append(
                    out.detach().float()))
        handles.append(h)
    return handles


def dist_from_logits(lg, family):
    """Free probability distribution per token, [T,E]."""
    if family == "lfm":
        s = lg.sigmoid()
        return s / s.sum(-1, keepdim=True)
    return torch.softmax(lg, -1)


def masked_dist(lg, R, family):
    """Residency-masked, renormalised distribution from the same logits."""
    mask = GL.compute_resident_mask_accel(lg.unsqueeze(1), R, evict="min_logit").squeeze(1)
    p = dist_from_logits(lg, family) * mask
    return p / p.sum(-1, keepdim=True).clamp_min(1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("gemma4", "qwen35", "lfm"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--R", type=int, required=True, help="residency budget (=k)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(ABLATIONS, "router_wasserstein.csv"))
    A = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        A.model_path, dtype=torch.bfloat16).to("cuda").eval()
    patch = {"gemma4": GL.patch_gemma4, "qwen35": GL.patch_qwen35,
             "lfm": GL.patch_lfm}[A.family]
    patch()
    tag = {"gemma4": GL.tag_gemma4, "qwen35": GL.tag_qwen35}.get(A.family)
    if tag:
        tag(model)
    routers = routers_of(model, A.family)
    print(f"[rw] {A.family}: {len(routers)} MoE layers", flush=True)

    prompts = [json.loads(l)["text"] for l in open(PROMPTS)][: A.n]
    L = len(routers)
    sums_imp = torch.zeros(L)
    sums_e2e = torch.zeros(L)
    ntok = 0
    for pi, text in enumerate(prompts):
        msgs = [{"role": "user", "content": text}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=True, return_dict=True)
        ids = torch.tensor([enc["input_ids"][: A.max_tok]], device="cuda")
        free_store, con_store = {}, {}
        with torch.no_grad():
            GL.CFG.update(on=False, enforce_from=0, batch=1)
            hs = hook_capture(routers, A.family, free_store)
            model(ids)
            for h in hs:
                h.remove()
            GL.CFG.update(on=True, R=A.R, enforce_from=0, batch=1,
                          cold_start=False)
            hs = hook_capture(routers, A.family, con_store)
            model(ids)
            for h in hs:
                h.remove()
            GL.CFG.update(on=False, batch=1)
        T = ids.shape[1]
        for li in range(L):
            lf = free_store[li][0].reshape(T, -1)
            lc = con_store[li][0].reshape(T, -1)
            pf = dist_from_logits(lf, A.family)
            q_imp = masked_dist(lf, A.R, A.family)
            q_e2e = masked_dist(lc, A.R, A.family)
            sums_imp[li] += 0.5 * (pf - q_imp).abs().sum(-1).sum().cpu()
            sums_e2e[li] += 0.5 * (pf - q_e2e).abs().sum(-1).sum().cpu()
        ntok += T
        if pi % 25 == 0:
            print(f"[rw] {pi}/{len(prompts)}", flush=True)

    new = not os.path.exists(A.out)
    with open(A.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["# per-layer W1 (unit ground metric = total variation) between free and "
                        "residency-constrained router distributions, base IT checkpoints, "
                        "teacher-forced WildChat (chat template, constraint from position 0, R=k). "
                        "w1_imposed: mask on free logits (no compounding); w1_endtoend: constrained "
                        "forward (with drift). lfm distributions are normalised sigmoid scores. "
                        "Producer: analysis/residency/router_wasserstein.py"])
            w.writerow(["model", "family", "R", "layer", "w1_imposed", "w1_endtoend", "n_tokens"])
        for li in range(L):
            w.writerow([os.path.basename(A.model_path.rstrip("/")), A.family, A.R, li,
                        f"{sums_imp[li].item()/ntok:.5f}",
                        f"{sums_e2e[li].item()/ntok:.5f}", ntok])
    print(f"[rw] DONE {A.family}: mean imposed "
          f"{sums_imp.sum().item()/ntok/L:.4f}, endtoend "
          f"{sums_e2e.sum().item()/ntok/L:.4f} over {ntok} tokens", flush=True)


if __name__ == "__main__":
    main()
