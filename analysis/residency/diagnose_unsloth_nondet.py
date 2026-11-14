#!/usr/bin/env python3
"""Locate the source of unsloth's gemma4 non-determinism.

Observed (2026-08-25, this pod): six identical forwards of one probe through an
unsloth-loaded gemma4 -- no residency patch, no expert-LoRA, constraint off -- differ
pairwise by 5.75 to 10.75 max|dlogit|. Post-warmup #1 vs #5 is 9.00, so it is not compile
or autotune warmup. Divergence starts at the FIRST transformer block (layer 0 identical,
layer 1 = 0.25) and compounds monotonically to 31.6 by layer 30. Plain HF on the same
weights and machine is bit-exact (0.000000). This forced both adapters onto --no-unsloth,
giving up unsloth's fused expert path.

Leading hypothesis: non-deterministic REDUCTION in the MoE combine. Scattering expert
outputs back with atomics (index_add_/scatter_add_ on CUDA) makes float addition order
vary run to run. That would appear at the first MoE layer and compound with depth, which
matches the observed profile exactly.

Three checks, cheapest first:

  1. deterministic-algorithms probe. torch.use_deterministic_algorithms(True) either makes
     the forward reproducible -- confirming a non-deterministic kernel -- or raises naming
     the offending op outright, which is the answer for free.
  2. component bisect. Within the first block, compare attention output and MoE output
     separately across two runs, to say which half is unstable.
  3. isolate the MoE combine. Run the block's MoE twice on identical input.

    diagnose_unsloth_nondet.py --model /dev/shm/gemma4-26b-it
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def probe_rows(path, n=256):
    rows = torch.load("/workspace/instruct-traj/gemma4_d7_seq4096.pt",
                      weights_only=False)["rows"]
    return rows[0]["ids"][:n].to("cuda").long().unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/gemma4-26b-it")
    ap.add_argument("--runs", type=int, default=3)
    A = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)

    print("=== 1. deterministic-algorithms probe ===", flush=True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        det_on = True
    except Exception as e:
        print(f"  could not enable: {e}"); det_on = False

    from unsloth import FastModel
    model, _ = FastModel.from_pretrained(A.model, max_seq_length=4096,
                                         dtype=torch.bfloat16, load_in_4bit=False,
                                         full_finetuning=False)
    model.eval()
    probe = probe_rows(A.model)

    def forward_logits():
        with torch.no_grad():
            return model(probe).logits[:, -1].float()

    try:
        outs = [forward_logits() for _ in range(A.runs)]
        d = max(float((outs[0] - o).abs().max()) for o in outs[1:])
        print(f"  deterministic_algorithms={det_on}: max|dlogit| across {A.runs} runs = {d:.6f}")
        print("  VERDICT:", "reproducible -> a non-deterministic kernel was the cause"
              if d == 0 else "STILL non-deterministic -> not a flagged op; bisect below")
    except RuntimeError as e:
        # the useful failure: torch names the offending op
        print(f"  RuntimeError names the culprit:\n    {str(e)[:400]}")
        return

    print("\n=== 2. per-layer divergence profile ===", flush=True)
    with torch.no_grad():
        a = model(probe, output_hidden_states=True).hidden_states
        b = model(probe, output_hidden_states=True).hidden_states
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        dd = float((x.float() - y.float()).abs().max())
        if dd > 0 and first is None:
            first = i
        if i < 3 or (first is not None and i <= first + 2) or i == len(a) - 1:
            print(f"    layer {i:2d}: max|dh| = {dd:.6f}")
    print(f"    first diverging layer: {first}")


if __name__ == "__main__":
    main()
