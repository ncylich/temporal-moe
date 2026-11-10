#!/usr/bin/env python3
"""Step 3 check: engine logits from a GGUF versus the PyTorch reference on the same token ids.

Both sides use unconstrained top-k routing with gates renormalised over the selected experts
(the reference's `topk` mode, the engine's stock router). Reports the max-abs logit difference
and the per-token loss difference. Top-k selection is discontinuous, so a near-tie between the
k-th and (k+1)-th expert can resolve differently under float rounding and move a few positions
by far more than the rest; the per-position summary separates those from a systematic mismatch.

    python scripts/serving_convert/compare_gguf.py --weights $WORK/fine_1e19.safetensors \
        --gguf $WORK/fine_1e19-f16.gguf --bin $WORK/gguf_logits \
        --tokens $WORK/test_tokens.npz --samples 4 --seq 2048
"""
import argparse, os, subprocess, sys, tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_model import RefModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--bin", required=True, help="compiled gguf_logits")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--control", action="store_true",
                    help="also diff the float32 reference against a float64 reference, to show "
                         "what rounding alone does through top-k near-ties")
    ap.add_argument("--flash-attn", action="store_true",
                    help="run the engine with flash attention (its default; K/V cast to F16)")
    a = ap.parse_args()

    toks = np.load(a.tokens)["tokens"][:a.samples, :a.seq + 1].astype(np.int64)
    model = RefModel(a.weights, a.topk)
    tmp = tempfile.mkdtemp()
    tot_ref, tot_gg, worst, pos_max = [], [], 0.0, []
    for i, row in enumerate(toks):
        ref = model.forward(torch.from_numpy(row[None, :-1]), "topk")[0]
        if a.control:
            r64 = model.forward(torch.from_numpy(row[None, :-1]), "topk", double=True)[0]
            c = (ref.double() - r64).abs().max(-1).values
            print(f"sample {i}: control ref32 vs ref64  max|dlogit| {c.max():.2e}  "
                  f"median {c.median():.2e}  positions above 1e-3: {(c > 1e-3).sum().item()}")
        row[:-1].astype(np.int32).tofile(f"{tmp}/t.i32")
        subprocess.run([a.bin, "logits", a.gguf, f"{tmp}/t.i32", str(a.seq), f"{tmp}/o.f32"]
                       + (["fa"] if a.flash_attn else []),
                       check=True, stderr=subprocess.DEVNULL)
        gg = torch.from_numpy(np.fromfile(f"{tmp}/o.f32", dtype=np.float32).reshape(a.seq, -1))
        tgt = torch.from_numpy(row[1:])
        lr = F.cross_entropy(ref, tgt, reduction="none")
        lg = F.cross_entropy(gg, tgt, reduction="none")
        d = (ref - gg).abs()
        worst = max(worst, d.max().item())
        pos_max.append(d.max(-1).values)
        tot_ref.append(lr); tot_gg.append(lg)
        print(f"sample {i}: ref loss {lr.mean():.5f}  gguf loss {lg.mean():.5f}  "
              f"max|dlogit| {d.max():.2e}  mean|dlogit| {d.mean():.2e}  "
              f"max|dloss| {(lr-lg).abs().max():.4f}  argmax agree {(ref.argmax(-1)==gg.argmax(-1)).float().mean():.4f}",
              flush=True)
    lr, lg = torch.cat(tot_ref), torch.cat(tot_gg)
    print(f"TOTAL tokens {lr.numel()}: ref loss {lr.mean():.5f}  gguf loss {lg.mean():.5f}  "
          f"diff {lg.mean()-lr.mean():+.5f}  max|dlogit| {worst:.2e}  "
          f"median|dloss| {(lr-lg).abs().median():.5f}")
    pm = torch.cat(pos_max)
    print(f"per-position max|dlogit|: median {pm.median():.2e}  p99 {pm.quantile(0.99):.2e}  "
          f"positions above 1e-3: {(pm > 1e-3).sum().item()} of {pm.numel()}")


if __name__ == "__main__":
    main()
