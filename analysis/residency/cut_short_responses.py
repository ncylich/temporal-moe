#!/usr/bin/env python3
"""Keep only SHORT-response trajectory rows -- a repackaging of existing data.

Hypothesis. The published recipe's 3.4M response tokens spanned 9,173 rows, i.e. a mean of
371 tokens per response. Our trajectories average 668-868. Long, meandering solutions range
over many experts; short direct ones can stay inside a small resident set. At R16 (16 of 128
experts) there is room either way; at R8 (8 of 128) there is not -- which is exactly the
observed shape, R16 undamaged in every arm and R8 failing in every arm, and it also explains
why DOUBLING the budget made R8 worse rather than better.

This generates no new data. It selects a subset of trajectories the model already produced,
so the lineage position is unchanged.

    cut_short_responses.py --tag gemma4_d7_seq4096 --max-resp 640
"""
import argparse
import json
import os

import torch

TRAJ = "/workspace/instruct-traj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-resp", type=int, default=640)
    A = ap.parse_args()
    blob = torch.load(f"{TRAJ}/{A.tag}.pt", weights_only=False)
    rows = blob["rows"]
    keep = [r for r in rows if (len(r["ids"]) - int(r["prompt_len"])) <= A.max_resp]
    toks = sum(len(r["ids"]) - int(r["prompt_len"]) for r in keep)
    out = f"{TRAJ}/{A.tag}_short{A.max_resp}.pt"
    torch.save({"rows": keep, "meta": dict(blob.get("meta") or {},
                short_cut=A.max_resp, from_tag=A.tag)}, out)
    meta = {"tag": f"{A.tag}_short{A.max_resp}", "rows_in": len(rows),
            "rows_kept": len(keep), "max_resp": A.max_resp,
            "response_tokens": toks, "mean_resp": round(toks / max(1, len(keep)), 1),
            "published_mean_resp": 371}
    with open(out.replace(".pt", ".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
