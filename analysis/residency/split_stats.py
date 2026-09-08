#!/usr/bin/env python3
"""Per-expert channel-partition statistics for the informed half-grain split.

Teacher-forces WildChat prompts through the ORIGINAL checkpoint (free routing),
captures each experts-module call (input x, routed top-k indices), and for every
expert recomputes its intermediate activations h = act(W_gate x) * (W_up x) on
the tokens routed to it. Accumulates per expert:

  sketch S = sum_t h_t (h_t^T Omega)   (d x K randomized covariance sketch)
  absmean  = sum_t |h_t|, and token count

The partition is the balanced median split along the top eigenvector of the
sketched covariance (one-pass randomized range finder): channels that co-fire
land in the same half, so the two halves activate on DIFFERENT inputs and are
functionally differentiated from birth. Saves per-layer, per-expert channel
permutations (argsort of the eigenvector) to an npz consumed by
split_experts.py --partition.

    split_stats.py --family qwen35 --model-path ... --out parts.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functional_displacement import experts_of                        # noqa: E402
from router_wasserstein import routers_of, hook_capture               # noqa: E402

PROMPTS = "/workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl"
SKETCH_K = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("gemma4", "qwen35"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        A.model_path, dtype=torch.bfloat16).to("cuda").eval()
    experts = experts_of(model, A.family)
    routers = routers_of(model, A.family)
    L = len(experts)
    E = experts[0].num_experts
    D = experts[0].intermediate_dim
    act = experts[0].act_fn
    print(f"[ss] {A.family}: {L} layers, E={E}, d={D}", flush=True)

    g = torch.Generator(device="cuda").manual_seed(0)
    omega = torch.randn(D, SKETCH_K, generator=g, device="cuda")
    S = torch.zeros(L, E, D, SKETCH_K, device="cuda")          # 40*256*512*8*4B = 84MB
    AM = torch.zeros(L, E, D, device="cuda")
    CNT = torch.zeros(L, E, device="cuda")

    prompts = [json.loads(l)["text"] for l in open(PROMPTS)][: A.n]
    for pi, text in enumerate(prompts):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True,
                                      tokenize=True, return_dict=True)
        ids = torch.tensor([enc["input_ids"][: A.max_tok]], device="cuda")
        xstore, wstore = {}, {}

        def _pre(_m, args, li):
            xstore[li] = args[0].detach()
            wstore[li] = args[1].detach()

        hs = [ex.register_forward_pre_hook(
                  lambda _m, args, li=li: _pre(_m, args, li))
              for li, ex in enumerate(experts)]
        with torch.no_grad():
            model(ids)
        for h in hs:
            h.remove()

        with torch.no_grad():
            for li in range(L):
                x = xstore[li].reshape(-1, xstore[li].shape[-1])
                idx = wstore[li].reshape(x.shape[0], -1)
                ex = experts[li]
                flat_e = idx.reshape(-1)
                tok_i = torch.arange(x.shape[0], device=x.device
                                     ).repeat_interleave(idx.shape[1])
                keep = flat_e < E
                flat_e, tok_i = flat_e[keep], tok_i[keep]
                order = torch.argsort(flat_e)
                flat_e, tok_i = flat_e[order], tok_i[order]
                counts = torch.bincount(flat_e, minlength=E)
                off = 0
                for e in counts.nonzero().flatten().tolist():
                    c = int(counts[e])
                    xt = x[tok_i[off:off + c]]
                    off += c
                    gu = torch.nn.functional.linear(xt, ex.gate_up_proj[e])
                    gate, up = gu.chunk(2, dim=-1)
                    hcH = (act(gate) * up).float()
                    S[li, e] += hcH.T @ (hcH @ omega)
                    AM[li, e] += hcH.abs().sum(0)
                    CNT[li, e] += c
        if pi % 25 == 0:
            print(f"[ss] {pi}/{len(prompts)}", flush=True)

    perms = np.zeros((L, E, D), dtype=np.int32)
    weak = 0
    for li in range(L):
        for e in range(E):
            if CNT[li, e] < 32:                      # too few tokens: importance order
                v = AM[li, e]
                weak += 1
            else:
                # top eigenvector from the sketch: orthonormalize S, then power step
                Q, _ = torch.linalg.qr(S[li, e])
                B = S[li, e].T @ Q                   # K x K proxy
                _, _, Vh = torch.linalg.svd(B)
                v = (Q @ Vh[0])                      # principal co-activation axis
                if v.sum() < 0:
                    v = -v
            perms[li, e] = torch.argsort(v, descending=True).cpu().numpy()
    np.savez(A.out, perms=perms, n_tokens=int(CNT.sum().item()),
             weak_experts=weak, sketch_k=SKETCH_K)
    print(f"[ss] DONE: perms {perms.shape}, {weak} low-count experts fell back "
          f"to abs-mean ordering, {int(CNT.sum())} routed tokens", flush=True)


if __name__ == "__main__":
    main()
