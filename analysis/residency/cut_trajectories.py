#!/usr/bin/env python3
"""Drop over-length trajectory rows WHOLE, before training reads them.

The recipe's own rule: "Training rows must never be truncated -- mid-response cuts teach
degenerate early endings (7 to 13-token IFEval answers); gates drop over-length rows
whole." The original trajectories were generated at a 2048 cap, so every row fitted the
training sequence and no gate was needed -- and train_gemma_ce.py accordingly has no
length filter at all (train_qwen_ce.py does). These trajectories were deliberately
generated at 8192 so the cut could be chosen from the measured distribution instead of
guessed, which means the gate now has to exist.

Cut is on TOTAL length (prompt + response) against the training sequence length, not on
response length alone: a long response after a short prompt still trains fine, and rows
that hit the generation cap are truncated mid-thought and get dropped by the same test.

    cut_trajectories.py --tag gemma4_d7 --max-seq 4096
"""
import argparse
import json
import os

import torch

TRAJ = "/workspace/instruct-traj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--gen-cap", type=int, default=8192,
                    help="cap the trajectories were generated at; rows at it were "
                         "truncated mid-response and are reported separately")
    A = ap.parse_args()

    src = f"{TRAJ}/{A.tag}.pt"
    blob = torch.load(src, weights_only=False)
    rows = blob["rows"]
    keep, capped = [], 0
    for r in rows:
        resp = len(r["ids"]) - int(r["prompt_len"])
        if resp >= A.gen_cap - 1:
            capped += 1
        if len(r["ids"]) <= A.max_seq:
            keep.append(r)

    toks = sum(len(r["ids"]) - int(r["prompt_len"]) for r in keep)
    dropped = len(rows) - len(keep)
    out = f"{TRAJ}/{A.tag}_seq{A.max_seq}.pt"
    torch.save({"rows": keep, "meta": dict(blob.get("meta") or {},
                cut_from=src, max_seq=A.max_seq, dropped=dropped,
                truncated_at_gen_cap=capped)}, out)
    meta = {"tag": f"{A.tag}_seq{A.max_seq}", "src": src, "max_seq": A.max_seq,
            "rows_in": len(rows), "rows_kept": len(keep), "rows_dropped": dropped,
            "drop_pct": round(100 * dropped / len(rows), 2),
            "truncated_at_gen_cap": capped,
            "response_tokens_kept": toks}
    with open(f"{TRAJ}/{A.tag}_seq{A.max_seq}.meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[cut] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
