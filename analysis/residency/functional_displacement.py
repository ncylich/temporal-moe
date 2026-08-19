#!/usr/bin/env python3
"""Per-layer FUNCTIONAL displacement under residency, plus aggregate expert-usage
shift. Companion to router_wasserstein.py, which showed that probability-space
displacement does not predict damage; this measures what the substitution does
to the layer's computation.

For each MoE layer, teacher-forced WildChat prompts (free forward), we capture
the experts module's input x, its free output y_free, and the router logits.
Offline per layer we build the residency mask from the free logits (same scan
and convention as the W1 run: R=k, min-logit, imposed from position 0), rebuild
the top-k routing restricted to residents with each family's exact routing
semantics, and re-execute the experts module on the same x:

  rel_out   sum_t ||y_masked - y_free||_2 / sum_t ||y_free||_2
  cos       mean_t cosine(y_masked, y_free)
  usage_tv  TV between dataset-aggregate expert-usage histograms
            (mean free full softmax vs mean masked-renormalised distribution)

Shared experts (qwen) are outside the experts module and identical in both
arms, so the comparison is routed-experts only. Every family's routing
reconstruction is verified against the model's own free forward per prompt
(top-k sets and weights from captured logits must match the captured ones).

    functional_displacement.py --family qwen35 --model-path ... --R 8
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import granularity_ladder as GL                                      # noqa: E402
from paths import ABLATIONS                                          # noqa: E402
from router_wasserstein import routers_of, hook_capture, dist_from_logits  # noqa: E402

PROMPTS = "/workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl"


def experts_of(model, family):
    if family == "gemma4":
        from transformers.models.gemma4 import modeling_gemma4 as m
        cls = m.Gemma4TextExperts
    elif family == "qwen35":
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
        cls = m.Qwen3_5MoeExperts
    elif family == "lfm":
        from transformers.models.lfm2_moe import modeling_lfm2_moe as m
        cls = m.Lfm2MoeExperts
    elif family == "olmoe":
        from transformers.models.olmoe import modeling_olmoe as m
        cls = m.OlmoeExperts
    elif family == "gptoss":
        from transformers.models.gpt_oss import modeling_gpt_oss as m
        cls = m.GptOssExperts
    else:
        raise ValueError(family)
    return [mod for mod in model.modules() if isinstance(mod, cls)]


def route_from_logits(lg, mask, family, router, block):
    """Rebuild (top_k_index, top_k_weights) from captured float logits with the
    family's exact routing semantics, restricted to `mask` (bool [T,E]; pass
    all-True to reproduce free routing for verification)."""
    if family == "gemma4":
        probs = torch.softmax(lg, -1) * mask
        w, idx = torch.topk(probs, router.config.top_k_experts, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        w = w * router.per_expert_scale[idx]
        return idx, w
    if family in ("qwen35", "olmoe"):
        probs = torch.softmax(lg, dtype=torch.float, dim=-1) * mask
        w, idx = torch.topk(probs, router.top_k, dim=-1)
        if family == "qwen35" or router.norm_topk_prob:
            w = w / w.sum(-1, keepdim=True)
        return idx, w
    if family == "gptoss":
        w, idx = torch.topk(lg.masked_fill(~mask, float("-inf")), router.top_k, dim=-1)
        w = torch.softmax(w, dim=1)
        return idx, w
    if family == "lfm":
        rw = lg.sigmoid()
        if block.use_expert_bias:
            scores = (rw + block.expert_bias).masked_fill(~mask, float("-inf"))
            _, idx = torch.topk(scores, k=block.top_k, dim=-1)
            w = torch.gather(rw, 1, idx)
        else:
            w, idx = torch.topk(rw.masked_fill(~mask, 0.0), k=block.top_k, dim=-1)
        if block.norm_topk_prob:
            w = w / (w.sum(-1, keepdim=True) + 1e-6)
        return idx, w * block.routed_scaling_factor
    raise ValueError(family)


def blocks_of(model, family):
    """Module holding routing config needed beyond the router (lfm only)."""
    if family != "lfm":
        return None
    from transformers.models.lfm2_moe import modeling_lfm2_moe as m
    return [mod for mod in model.modules() if isinstance(mod, m.Lfm2MoeSparseMoeBlock)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=("gemma4", "qwen35", "lfm", "olmoe", "gptoss"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dequantize", action="store_true",
                    help="gpt-oss: dequantize MXFP4 to bf16 (quantized load swaps "
                         "out GptOssExperts, breaking module discovery)")
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(ABLATIONS, "functional_displacement.csv"))
    A = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model_path)
    kw = {"device_map": "cuda"} if A.family == "gptoss" else {}
    if A.family == "gptoss" and A.dequantize:
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(A.model_path, dtype=torch.bfloat16, **kw)
    if not kw:
        model = model.to("cuda")
    model = model.eval()
    routers = routers_of(model, A.family)
    experts = experts_of(model, A.family)
    blocks = blocks_of(model, A.family)
    assert len(routers) == len(experts), (len(routers), len(experts))
    L = len(routers)
    print(f"[fd] {A.family}: {L} MoE layers", flush=True)

    prompts = [json.loads(l)["text"] for l in open(PROMPTS)][: A.n]
    sums = {k: torch.zeros(L, dtype=torch.float64) for k in
            ("dnorm", "ynorm", "xnorm", "cos")}
    usage_free = usage_mask = None
    ntok = 0
    flip_tok = 0   # tokens whose reconstructed free top-k set != model's own
    flip_den = 0   # (bf16-vs-float32 ties at the top-k boundary; should be <~1%)

    for pi, text in enumerate(prompts):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True,
                                      tokenize=True, return_dict=True)
        ids = torch.tensor([enc["input_ids"][: A.max_tok]], device="cuda")
        T = ids.shape[1]
        lstore, xstore, ystore, wstore = {}, {}, {}, {}
        hs = hook_capture(routers, A.family, lstore)

        def _pre(_m, args, li):
            xstore[li] = args[0].detach()
            wstore[li] = (args[1].detach(), args[2].detach())

        def _post(_m, _i, out, li):
            ystore[li] = out.detach()

        pre = [ex.register_forward_pre_hook(
                   lambda _m, args, li=li: _pre(_m, args, li))
               for li, ex in enumerate(experts)]
        post = [ex.register_forward_hook(
                    lambda _m, _i, out, li=li: _post(_m, _i, out, li))
                for li, ex in enumerate(experts)]
        with torch.no_grad():
            model(ids)
        for h in hs + pre + post:
            h.remove()

        with torch.no_grad():
            for li in range(L):
                lg = lstore[li][0].reshape(T, -1)
                x = xstore[li].reshape(T, -1)
                y_f = ystore[li].reshape(T, -1).float()
                idx_f, w_f = wstore[li]
                idx_f, w_f = idx_f.reshape(T, -1), w_f.reshape(T, -1)
                if usage_free is None and li == 0:
                    nE = lg.shape[-1]
                    usage_free = torch.zeros(L, nE, dtype=torch.float64, device="cuda")
                    usage_mask = torch.zeros(L, nE, dtype=torch.float64, device="cuda")
                full = torch.ones_like(lg, dtype=torch.bool)
                idx_v, w_v = route_from_logits(
                    lg, full, A.family, routers[li], blocks[li] if blocks else None)
                same = (idx_v.sort(-1).values
                        == idx_f.sort(-1).values.to(idx_v.dtype)).all(-1)
                flip_tok += (~same).sum().item()
                flip_den += T
                # weight check only where the top-k set matched; tie-flipped
                # tokens legitimately carry a different weight (e.g. lfm gathers
                # unbiased sigmoids after biased selection)
                assert torch.allclose(w_v.float().sort(-1).values[same],
                                      w_f.float().sort(-1).values[same], atol=0.05), \
                    f"weight reconstruction off at layer {li}"
                mask = GL.compute_resident_mask_accel(
                    lg.unsqueeze(1), A.R, evict="min_logit").squeeze(1).bool()
                idx_m, w_m = route_from_logits(
                    lg, mask, A.family, routers[li], blocks[li] if blocks else None)
                y_m = experts[li](x, idx_m, w_m.to(x.dtype)).float()
                d = (y_m - y_f).norm(dim=-1)
                sums["dnorm"][li] += d.sum().double().cpu()
                sums["ynorm"][li] += y_f.norm(dim=-1).sum().double().cpu()
                sums["xnorm"][li] += x.float().norm(dim=-1).sum().double().cpu()
                sums["cos"][li] += F.cosine_similarity(y_m, y_f, dim=-1).sum().double().cpu()
                pf = dist_from_logits(lg, A.family)
                q = pf * mask
                q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
                usage_free[li] += pf.sum(0).double()
                usage_mask[li] += q.sum(0).double()
        ntok += T
        if pi % 25 == 0:
            print(f"[fd] {pi}/{len(prompts)} (top-k tie-flip rate so far: "
                  f"{flip_tok/max(flip_den,1):.4%})", flush=True)

    uf = (usage_free / ntok).cpu().numpy()
    um = (usage_mask / ntok).cpu().numpy()
    usage_tv = 0.5 * np.abs(uf - um).sum(-1)

    name = os.path.basename(A.model_path.rstrip("/"))
    np.savez(os.path.join(ABLATIONS, f"functional_displacement_usage_{name}.npz"),
             usage_free=uf, usage_masked=um, R=A.R, n_tokens=ntok)
    new = not os.path.exists(A.out)
    with open(A.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["# Per-layer functional displacement under residency (imposed: mask from "
                        "the free forward's own logits, R=k min-logit from position 0, same scan "
                        "as router_wasserstein.csv). rel_out = sum||y_masked-y_free|| / "
                        "sum||y_free|| over the routed-experts module output on identical inputs; "
                        "cos = mean token cosine(y_masked, y_free); usage_tv = total variation "
                        "between dataset-mean free and masked-renormalised router distributions "
                        "(aggregate expert-usage shift). Teacher-forced WildChat, base IT "
                        "checkpoints. Producer: analysis/residency/functional_displacement.py"])
            w.writerow(["model", "family", "R", "layer", "rel_out", "cos", "usage_tv",
                        "rel_to_input", "n_tokens", "verify_tie_flip_rate"])
        for li in range(L):
            w.writerow([name, A.family, A.R, li,
                        f"{(sums['dnorm'][li]/sums['ynorm'][li]).item():.5f}",
                        f"{(sums['cos'][li]/ntok).item():.5f}",
                        f"{usage_tv[li]:.5f}",
                        f"{(sums['dnorm'][li]/sums['xnorm'][li]).item():.5f}",
                        ntok, f"{flip_tok/max(flip_den,1):.5f}"])
    print(f"[fd] DONE {A.family}: mean rel_out "
          f"{(sums['dnorm'].sum()/sums['ynorm'].sum()).item():.4f}, mean cos "
          f"{(sums['cos'].sum()/(ntok*L)).item():.4f}, mean usage_tv {usage_tv.mean():.4f}, "
          f"tie-flip rate {flip_tok/max(flip_den,1):.4%} over {ntok} tokens", flush=True)


if __name__ == "__main__":
    main()
