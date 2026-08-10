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

        def set_arm(R, plen):
            RES._CFG.update(on=R is not None, R=R or 0, evict="min_logit",
                            gate_mass="preserve", swaps=1, R_map=None,
                            collect_telem=False, enforce_from=plen)
            RES.set_free_layers(None)
    elif M["arch"] == "qwen3_5":
        import residency as RES
        import residency_qwen as RQ
        model, _ = RQ.load_model(M["path"], family="qwen3_5")

        def set_arm(R, plen):
            RES._CFG.update(on=R is not None, R=R or 0, evict="min_logit",
                            gate_mass="preserve", swaps=1, R_map=None,
                            collect_telem=False, enforce_from=plen, free_set=None)
    else:
        from transformers import AutoModelForCausalLM
        import granularity_ladder as GL
        model = AutoModelForCausalLM.from_pretrained(
            M["path"], dtype=torch.bfloat16).to("cuda")
        model.eval()
        {"lfm": GL.patch_lfm, "gemma4": GL.patch_gemma4}[M["arch"]]()

        def set_arm(R, plen):
            GL.CFG.update(on=R is not None, R=R or 0, free_set=None, R_map=None,
                          enforce_from=plen)
    return model, tok, set_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--n-traj", type=int, default=500)
    A = ap.parse_args()
    M = MODELS[A.model]

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
                 'Producer: analysis/ple/instruct_selfce.py"\n')
        w.writerow(["model", "E", "k", "arm", "R", "frac_pct", "ce_nats_per_tok",
                    "bits_per_byte", "resp_tokens", "resp_bytes", "n_traj", "secs"])

    resp_bytes = sum(len(tok.decode(r["ids"][r["prompt_len"]:].tolist(),
                                    skip_special_tokens=False).encode("utf-8"))
                     for r in rows)

    def arm(name, R):
        t0, tot, ntok = time.time(), 0.0, 0
        with torch.no_grad():
            for r in rows:
                ids = r["ids"].to("cuda").long().unsqueeze(0)
                plen = r["prompt_len"]
                set_arm(R, plen)
                lg = model(ids).logits[0].float()
                tot += float(torch.nn.functional.cross_entropy(
                    lg[plen - 1:-1], ids[0, plen:], reduction="sum"))
                ntok += ids.shape[1] - plen
                del lg
        ce = tot / ntok
        bpb = (tot / math.log(2)) / resp_bytes
        frac = "" if R is None else f"{100*R/M['E']:.2f}"
        w.writerow([A.model, M["E"], M["k"], name, R or "", frac, f"{ce:.6f}",
                    f"{bpb:.6f}", ntok, resp_bytes, len(rows), f"{time.time()-t0:.1f}"])
        fh.flush()
        print(f"  [{A.model}] {name:10} CE={ce:.4f} nats/tok bpb={bpb:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return ce

    free = arm("free", None)
    assert free < 1.0, f"free self-CE implausible ({free:.3f}) - wiring/template suspect"
    for R in M["cells"]:
        c = arm(f"R{R}", R)
        assert c > free - 1e-4, f"constrained beat free ({c} vs {free}) - masking not engaged"
    fh.close()
    print(f"SELFCE {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
