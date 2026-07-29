#!/usr/bin/env python3
"""Verify the post-MoE placement guarantee (PLE_PLAN.md §2, §13).

The plan settles placement on the grounds that PLE "must not perturb the input that produced"
the layer output, because feeding it into the MoE input would change routing decisions. This
script tests exactly that property, with a non-zero table so the branch is actually doing
something:

  layer 0 router logits   MUST be bitwise identical with and without PLE.
                          PLE[tok, 0] is added AFTER layer 0's MoE, so layer 0 routes on an
                          input PLE has never touched. This is the guarantee.

  layer >0 router logits  are EXPECTED to differ. PLE writes into the residual stream, so
                          layer l+1 reads a stream that includes PLE[tok, l]. Any residual-stream
                          contribution does this; it is not the pre-MoE failure mode the plan
                          rules out, which would change a layer's OWN routing.

Reporting both halves matters: "no routing changed anywhere" would actually mean the branch was
inert, and "layer 0 changed" would mean the placement is wrong.
"""

import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES     # noqa: E402
import ple as PLE           # noqa: E402


def main():
    torch.manual_seed(0)
    model, _ = RES.load_model()
    RES.enable_residency(R=8)
    model.eval()
    ids = torch.randint(0, 50304, (1, 256), device="cuda")

    def router_logits():
        with torch.no_grad():
            return [r.float().clone() for r in model(ids, output_router_logits=True).router_logits]

    PLE.uninstall()
    base_rl = router_logits()
    with torch.no_grad():
        base_logits = model(ids).logits.clone()

    ple = PLE.install(model, 32, device="cuda")
    zero_rl = router_logits()
    with torch.no_grad():
        zero_logits = model(ids).logits.clone()

    # make the branch genuinely active
    with torch.no_grad():
        ple.U.normal_(0, 0.02)
    act_rl = router_logits()
    with torch.no_grad():
        act_logits = model(ids).logits.clone()

    print("=== zero table (init): whole model must be bit-identical to no-PLE ===")
    print(f"  logits bitwise equal      : {bool((base_logits == zero_logits).all())}")
    print(f"  all router logits equal   : {all(bool((a == b).all()) for a, b in zip(base_rl, zero_rl))}")

    print("\n=== active table: layer 0 routing must be untouched, deeper layers may move ===")
    same = [bool((a == b).all()) for a, b in zip(base_rl, act_rl)]
    print(f"  layer 0 router logits bitwise equal : {same[0]}")
    n_diff = sum(1 for s in same[1:] if not s)
    print(f"  layers >0 that differ               : {n_diff} of {len(same)-1}")
    print(f"  output logits changed at all        : {bool((base_logits != act_logits).any())}")
    print(f"  max |logit diff| vs no-PLE          : {float((base_logits - act_logits).abs().max()):.4f}")
    print("\n  per-layer bitwise-equal flags:", "".join("1" if s else "0" for s in same))

    ok = (bool((base_logits == zero_logits).all()) and same[0]
          and bool((base_logits != act_logits).any()))
    print(f"\nPLACEMENT_CHECK {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
