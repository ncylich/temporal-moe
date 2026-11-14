#!/usr/bin/env python3
"""Offline partition screen + rotation oracle for half-grain splits.

Captures per-ORIGINAL-expert intermediate activations (h, d-dim) and outputs
(y = W_d h) on teacher-forced WildChat, by un-splitting the naive half-grain
checkpoint's expert tensors (contiguous split preserves channel order, so
halves 2i/2i+1 concatenate back exactly; down_proj halves carry a 2x factor).

Scores partition families by single-half reconstruction error
    err(P) = E_t ||y_t - 2 W_d[:,S] h_t[S]||^2 / E_t ||y_t||^2,  mean over S in {A,B}
(the factor 2 matches serving: a lone resident half carries the pair's full
renormalised weight). Families: naive contiguous, spectral median-split,
redundancy interleave (even-odd along the principal axis), contribution-snake,
random controls, and the ROTATION ORACLE: the best rank-d/2 linear bottleneck
of y (SVD truncation) - the LaRoSA-style upper bound no permutation can beat.
Also reports near-duplicate gate rows (exact u/d-plane gauge freedom).

    partition_screen.py --split-path /dev/shm/gemma4-halfgrain --family gemma4
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
from paths import ABLATIONS                                           # noqa: E402

PROMPTS = "/workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl"
NSAMP = 128           # h samples kept per original expert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path", required=True)
    ap.add_argument("--family", default="gemma4", choices=("gemma4", "qwen35"))
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(ABLATIONS, "partition_screen.csv"))
    ap.add_argument("--save-rotations", default=None,
                    help="npz path: per-expert eigenbasis of the h-covariance "
                         "(descending eigenvalue order), consumed by "
                         "split_experts.py --rotate for lossy rotated re-basing")
    A = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.split_path)
    model = AutoModelForCausalLM.from_pretrained(
        A.split_path, dtype=torch.bfloat16).to("cuda").eval()
    experts = experts_of(model, A.family)
    L, E2, Dh = len(experts), experts[0].num_experts, experts[0].intermediate_dim
    E, D = E2 // 2, Dh * 2
    act = experts[0].act_fn
    print(f"[ps] {L} layers, {E} original experts, d={D}", flush=True)

    # un-split weights per layer: gate/up [E, D, H] (concat halves), down [E, H, D]
    UG, UU, UD = [], [], []
    for ex in experts:
        gu = ex.gate_up_proj.data.cpu()      # keep un-split copies on CPU (~40G on GPU otherwise)
        g_h, u_h = gu[:, :Dh], gu[:, Dh:]
        g = torch.cat([g_h[0::2], g_h[1::2]], dim=1)     # [E, D, H]
        u = torch.cat([u_h[0::2], u_h[1::2]], dim=1)
        d_h = ex.down_proj.data.cpu()        # [2E, H, Dh], carries the 2x factor
        d = torch.cat([d_h[0::2], d_h[1::2]], dim=2) / 2.0
        UG.append(g); UU.append(u); UD.append(d)

    # capture: expert-module inputs + routed pair indices from the split forward
    H = {li: [[] for _ in range(E)] for li in range(L)}   # h samples per expert
    CH = torch.zeros(L, E, D, D)                          # streaming h-covariance (CPU)
    CN = torch.zeros(L, E)
    prompts = [json.loads(l)["text"] for l in open(PROMPTS)][: A.n]
    for pi, text in enumerate(prompts):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True,
                                      tokenize=True, return_dict=True)
        ids = torch.tensor([enc["input_ids"][: A.max_tok]], device="cuda")
        xs, ws = {}, {}

        def _pre(_m, args, li):
            xs[li] = args[0].detach()
            ws[li] = args[1].detach()
        hs = [ex.register_forward_pre_hook(lambda _m, a, li=li: _pre(_m, a, li))
              for li, ex in enumerate(experts)]
        with torch.no_grad():
            model(ids)
        for h_ in hs:
            h_.remove()
        with torch.no_grad():
            for li in range(L):
                x = xs[li].reshape(-1, xs[li].shape[-1])
                orig_e = torch.unique(ws[li].reshape(-1)[ws[li].reshape(-1) < E2] // 2)
                for e in orig_e.tolist():
                    sel = ((ws[li].reshape(x.shape[0], -1) // 2) == e).any(-1)
                    xt = x[sel][:24].cpu().float()
                    g = torch.nn.functional.linear(xt, UG[li][e].float())
                    u = torch.nn.functional.linear(xt, UU[li][e].float())
                    hset = (act(g) * u).float()
                    # covariance accumulates UNCAPPED (oracle needs full rank);
                    # permutation scoring keeps a bounded sample
                    CH[li, e] += hset.T @ hset
                    CN[li, e] += hset.shape[0]
                    if len(H[li][e]) * 8 < NSAMP:
                        H[li][e].append(hset[:8])
        if pi % 20 == 0:
            print(f"[ps] {pi}/{len(prompts)}", flush=True)

    torch.manual_seed(0)
    fams = ["naive", "spectral", "interleave", "snake", "rand1", "rand2", "oracle"]
    errs = {f: [] for f in fams}
    dup_pairs = 0
    for li in range(L):
        Wd = UD[li].float()   # CPU; per-expert slices moved to GPU below
        for e in range(E):
            if not H[li][e]:
                continue
            h = torch.cat(H[li][e]).cuda()               # [n, D]
            Wde = Wd[e].cuda()
            if h.shape[0] < 16:
                continue
            y = h @ Wde.T                              # [n, Hdim]
            ynorm = float((y ** 2).sum())
            if ynorm < 1e-8:
                continue
            # orderings
            c = h.abs().mean(0) * Wde.norm(dim=0)      # contribution
            hc = h - h.mean(0)
            _, _, V = torch.linalg.svd(hc, full_matrices=False)
            v1 = V[0]
            order_spec = torch.argsort(v1, descending=True)
            order_c = torch.argsort(c, descending=True)

            def err_of(idxA):
                tot = 0.0
                for S in (idxA, np.setdiff1d(np.arange(D), idxA)):
                    S = torch.as_tensor(S, device=h.device)
                    yh = 2.0 * (h[:, S] @ Wde[:, S].T)
                    tot += float(((y - yh) ** 2).sum()) / ynorm
                return tot / 2

            errs["naive"].append(err_of(np.arange(D // 2)))
            errs["spectral"].append(err_of(order_spec[: D // 2].cpu().numpy()))
            errs["interleave"].append(err_of(order_spec[0::2].cpu().numpy()))
            snake = torch.cat([order_c[0::4], order_c[3::4]])
            errs["snake"].append(err_of(snake.cpu().numpy()[: D // 2]))
            for i, f in enumerate(("rand1", "rand2")):
                errs[f].append(err_of(torch.randperm(D)[: D // 2].cpu().numpy()))
            # rotation oracle: expected rank-D/2 bottleneck error from the FULL
            # streaming covariance (sample-count-independent): eigen mass of
            # Ch^{1/2} G Ch^{1/2}, G = Wd^T Wd
            Ch = (CH[li, e] / max(float(CN[li, e]), 1.0)).cuda()
            ev, Q = torch.linalg.eigh(Ch)
            ev = ev.clamp_min(0)
            Chalf = Q @ torch.diag(ev.sqrt()) @ Q.T
            G = Wde.T @ Wde
            M = Chalf @ G @ Chalf
            lam = torch.linalg.eigvalsh(M).clamp_min(0)
            if float(CN[li, e]) >= 2 * D:   # full-rank covariance only: below
                # this the trailing eigen mass is rank-shadowed (reads ~0)
                errs["oracle"].append(float(lam[: D - D // 2].sum() / lam.sum().clamp_min(1e-12)))
            # near-duplicate gate rows (exact gauge freedom)
            gn = torch.nn.functional.normalize(UG[li][e].float().cuda(), dim=1)
            sim = (gn @ gn.T).fill_diagonal_(0)
            if float(sim.max()) > 0.99:
                dup_pairs += 1

    if A.save_rotations:
        R = np.zeros((L, E, D, D), dtype=np.float16)
        for li in range(L):
            for e in range(E):
                Ch = (CH[li, e] / max(float(CN[li, e]), 1.0)).cuda()
                ev, Q = torch.linalg.eigh(Ch)          # ascending
                R[li, e] = Q.flip(-1).T.cpu().numpy()  # rows = descending eigvecs
        np.savez(A.save_rotations, rotations=R)
        print(f"[ps] rotations saved -> {A.save_rotations}", flush=True)

    new = not os.path.exists(A.out)
    import csv
    with open(A.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["# Single-half reconstruction error by partition family (lower better; "
                        "oracle = best rank-d/2 linear bottleneck, unreachable by permutations). "
                        "Producer: analysis/residency/partition_screen.py"])
            w.writerow(["family", "model", "mean_err", "median_err", "n_experts"])
        for f in fams:
            a = np.array(errs[f])
            w.writerow([f, os.path.basename(A.split_path), f"{a.mean():.4f}",
                        f"{np.median(a):.4f}", len(a)])
    print("[ps] RESULTS (single-half rel err, lower better):")
    for f in fams:
        a = np.array(errs[f])
        print(f"  {f:10s} mean {a.mean():.4f}  median {np.median(a):.4f}  (n={len(a)})")
    print(f"[ps] near-duplicate gate-row experts: {dup_pairs}")


if __name__ == "__main__":
    main()
