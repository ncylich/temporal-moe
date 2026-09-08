#!/usr/bin/env python3
"""Self-CE of instruct models under the temporal residency rule, on their own frozen
free-trajectory responses (gen_trajectories.py).

Protocol: prefill positions are FREE (mask overridden all-True via enforce_from; the scan
still observes the prompt so the resident set is warmed), the rule is enforced from the
first response token, and CE is scored on response tokens only. Arms per model: free
reference, R = k (resident exactly the active-expert count), and R = 12.5% of E where
that differs from k. Batch=1 throughout; gate conventions are each model's own.

    instruct_selfce.py --model olmoe_instruct
"""
import argparse
import csv
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

TRAJ = "/workspace/instruct-traj"
IM = "/workspace/instruct-models"
MODELS = {
    "olmoe_instruct": {"path": f"{IM}/olmoe-0125-instruct", "E": 64, "k": 8,
                       "cells": [8], "arch": "olmoe"},
    "lfm25_instruct": {"path": f"{IM}/lfm25-8b-a1b", "E": 32, "k": 4,
                       "cells": [4], "arch": "lfm"},
    "gemma4_instruct": {"path": f"{IM}/gemma4-26b-it", "E": 128, "k": 8,
                        "cells": [8, 16], "arch": "gemma4"},
    "qwen35_instruct": {"path": f"{IM}/qwen35-35b-a3b-instruct", "E": 256, "k": 8,
                        "cells": [8, 32], "arch": "qwen3_5"},
    # half-grain splits (analysis/residency/split_experts.py): function-preserving
    # relabeling, 2E half-experts, k doubles; R in half-expert units.
    "qwen35_halfgrain": {"path": "/dev/shm/qwen35-halfgrain", "E": 512, "k": 16,
                         "cells": [96], "arch": "qwen3_5"},
    "gemma4_halfgrain": {"path": f"{IM}/gemma4-halfgrain", "E": 256, "k": 16,
                         "cells": [48], "arch": "gemma4"},
    # gpt-oss: generative benchmarks only (vLLM stack); no frozen-500 self-CE arm.
    "gptoss_20b": {"path": "/dev/shm/gpt-oss-20b", "E": 32, "k": 4,
                   "cells": [4], "arch": "gptoss"},
    "gptoss_120b": {"path": "/dev/shm/gpt-oss-120b", "E": 128, "k": 4,
                    "cells": [4, 16], "arch": "gptoss"},
}


def load(name, M):
    """Return (model, tokenizer, set_arm) where set_arm(R_or_None, prompt_len) configures
    the constraint for the next forward. Each arch reuses its audited patch path."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M["path"])
    if M["arch"] == "olmoe":
        import residency as RES
        model, _ = RES.load_model(M["path"])
        model.eval()

        def set_arm(R, plen, cold=False):
            RES._CFG.update(on=R is not None, R=R or 0, evict="min_logit",
                            gate_mass="preserve", swaps=1, R_map=None,
                            collect_telem=False, enforce_from=plen, cold_start=cold)
            RES.set_free_layers(None)
    elif M["arch"] == "qwen3_5":
        import residency as RES
        import residency_qwen as RQ
        model, _ = RQ.load_model(M["path"], family="qwen3_5")

        def set_arm(R, plen, cold=False):
            RES._CFG.update(on=R is not None, R=R or 0, evict="min_logit",
                            gate_mass="preserve", swaps=1, R_map=None,
                            collect_telem=False, enforce_from=plen, free_set=None,
                            cold_start=cold)
    else:
        from transformers import AutoModelForCausalLM
        import granularity_ladder as GL
        model = AutoModelForCausalLM.from_pretrained(
            M["path"], dtype=torch.bfloat16).to("cuda")
        model.eval()
        {"lfm": GL.patch_lfm, "gemma4": GL.patch_gemma4}[M["arch"]]()

        def set_arm(R, plen, cold=False):
            GL.CFG.update(on=R is not None, R=R or 0, free_set=None, R_map=None,
                          enforce_from=plen, cold_start=cold)
    return model, tok, set_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--n-traj", type=int, default=500)
    ap.add_argument("--path", default=None,
                    help="checkpoint dir override (big models staged on /dev/shm)")
    ap.add_argument("--cold", action="store_true",
                    help="run COLD decode arms only (scan blind to the prompt), rows named R{R}cold")
    A = ap.parse_args()
    M = MODELS[A.model]
    if A.path:
        M = dict(M, path=A.path)

    rows = torch.load(f"{TRAJ}/{A.model}.pt", weights_only=False)["rows"][: A.n_traj]
    assert rows, "no trajectories"
    model, tok, set_arm = load(A.model, M)

    out = os.path.join(ABLATIONS, "instruct_selfce.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        fh.write('"# Self-CE of instruct models on their own frozen greedy trajectories '
                 '(gen_trajectories.py, 500 WildChat prompts). Prefill free, temporal rule '
                 'enforced from first response token (scan warmed on prompt), CE on response '
                 'tokens only, batch=1, min_logit <=1 swap/token, model-own gate convention. '
                 'bits_per_byte over utf8 response bytes. '
                 'Producer: analysis/residency/instruct_selfce.py"\n')
        w.writerow(["model", "E", "k", "arm", "R", "frac_pct", "ce_nats_per_tok",
                    "bits_per_byte", "ce_think", "ce_answer", "think_tokens",
                    "answer_tokens", "resp_tokens", "resp_bytes", "n_traj", "secs"])

    resp_bytes = sum(len(tok.decode(r["ids"][r["prompt_len"]:].tolist(),
                                    skip_special_tokens=False).encode("utf-8"))
                     for r in rows)

    # Thinking-segment split (reasoning-default models): a response token is "think" from the
    # start of the response through the first </think> tag inclusive; with no closing tag but
    # an opening one (cap hit mid-thought) the whole response counts as think. Models whose
    # tokenizer has no dedicated think tokens get an all-answer mask.
    def _tid(s):
        enc = tok.encode(s, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None
    OPEN, CLOSE = _tid("<think>"), _tid("</think>")
    for r in rows:
        resp = r["ids"][r["prompt_len"]:]
        m = torch.zeros(len(resp), dtype=torch.bool)
        if CLOSE is not None:
            close = (resp == CLOSE).nonzero()
            if len(close):
                m[: int(close[0]) + 1] = True
            elif OPEN is not None and bool((resp == OPEN).any()):
                m[:] = True
        r["think_mask"] = m

    def arm(name, R):
        t0 = time.time()
        tot = tot_th = tot_an = 0.0
        ntok = n_th = n_an = 0
        with torch.no_grad():
            for r in rows:
                ids = r["ids"].to("cuda").long().unsqueeze(0)
                plen = r["prompt_len"]
                set_arm(R, plen, cold=A.cold)
                lg = model(ids).logits[0].float()
                losses = torch.nn.functional.cross_entropy(
                    lg[plen - 1:-1], ids[0, plen:], reduction="none")
                m = r["think_mask"].to(losses.device)
                tot += float(losses.sum())
                tot_th += float(losses[m].sum())
                tot_an += float(losses[~m].sum())
                ntok += losses.numel()
                n_th += int(m.sum())
                n_an += int((~m).sum())
                del lg, losses
        ce = tot / ntok
        bpb = (tot / math.log(2)) / resp_bytes
        frac = "" if R is None else f"{100*R/M['E']:.2f}"
        ce_th = f"{tot_th/n_th:.6f}" if n_th else ""
        ce_an = f"{tot_an/n_an:.6f}" if n_an else ""
        w.writerow([A.model, M["E"], M["k"], name, R or "", frac, f"{ce:.6f}",
                    f"{bpb:.6f}", ce_th, ce_an, n_th, n_an, ntok, resp_bytes,
                    len(rows), f"{time.time()-t0:.1f}"])
        fh.flush()
        print(f"  [{A.model}] {name:10} CE={ce:.4f} nats/tok bpb={bpb:.4f} "
              f"think={ce_th or '-'} answer={ce_an or '-'} ({time.time()-t0:.0f}s)", flush=True)
        return ce

    if A.cold:
        for R in M["cells"]:
            arm(f"R{R}cold", R)
    else:
        free = arm("free", None)
        assert free < 1.0, f"free self-CE implausible ({free:.3f}) - wiring/template suspect"
        for R in M["cells"]:
            c = arm(f"R{R}", R)
            assert c > free - 1e-4, f"constrained beat free ({c} vs {free}) - masking not engaged"
    fh.close()
    print(f"SELFCE {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
