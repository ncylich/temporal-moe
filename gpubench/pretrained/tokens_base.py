#!/usr/bin/env python3
"""Real-text token files for the 1e19 base models (no chat template). The context is the first N tokens of a
held-out test row (DCLM test split, test_tokens.npz); the decoded tokens are this server's own greedy
continuation, so each routing method replays the text it would produce. Same file format as
gpubench/released/serve_client.py generate: <out>/ctx<N>_<tag>.ctx (N int32) and .gen (n_gen int32).

    python3 tokens_base.py --url http://127.0.0.1:8093 --tokens test_tokens.npz --row 600 --ctx 1024 2048 4096 \
        --tag ours --out tokens/v1
"""
import argparse, json, os, struct, urllib.request
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--tokens", required=True)
    ap.add_argument("--row", type=int, required=True); ap.add_argument("--ctx", type=int, nargs="+", default=[1024])
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-gen", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = np.load(a.tokens)["tokens"]
    for ctx in a.ctx:
        row = T[a.row]
        if len(row) < ctx:                                   # longer contexts: concatenate following rows
            row = np.concatenate([T[a.row + j] for j in range((ctx + len(row) - 1) // len(row))])
        prompt = [int(x) for x in row[:ctx]]
        body = {"prompt": prompt, "n_predict": a.n_gen, "temperature": 0.0, "cache_prompt": False,
                "return_tokens": True, "ignore_eos": True}
        req = urllib.request.Request(a.url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        gen = r["tokens"][:a.n_gen]
        assert len(gen) == a.n_gen, len(gen)
        for ext, part in (("ctx", prompt), ("gen", gen)):
            with open(os.path.join(a.out, f"ctx{ctx}_{a.tag}.{ext}"), "wb") as f:
                f.write(struct.pack(f"<{len(part)}i", *part))
        with open(os.path.join(a.out, f"ctx{ctx}_{a.tag}.txt"), "w") as f:
            f.write(r["content"])
        print(f"ctx {ctx} [{a.tag}] row {a.row}: wrote {ctx} + {a.n_gen}", flush=True)


if __name__ == "__main__":
    main()
