#!/usr/bin/env python3
"""Assert a merged checkpoint actually received EVERY trained surface.

Written after a silent failure on 2026-08-25: peft attached the attention LoRA to
gemma4's vision tower instead of its language model (only the vision tower wraps its
projections in Gemma4ClippableLinear), so a text-only forward never reached those
modules, every lora_B stayed at zero init, and the merge produced a checkpoint carrying
expert-LoRA and NO attention LoRA. Training completed, the smoke passed, and the merge
reported success. The only thing that showed it was diffing merged against base and
finding self_attn.q_proj.weight identical to 0.000000 while the expert tensors had moved.

So: never trust a merge that "succeeded". Diff it.

    verify_merge.py --base /dev/shm/gemma4-26b-it --merged /dev/shm/gemma4-rebuild-merged
"""
import argparse
import json
import sys

from safetensors import safe_open

# (label, substring, must_change) -- embeddings are NOT trained, so they are the control:
# if they moved, something merged that should not have.
SURFACES = [("expert gate_up", "experts.gate_up_proj", True),
            ("expert down", "experts.down_proj", True),
            ("attention q_proj", "self_attn.q_proj", True),
            ("attention o_proj", "self_attn.o_proj", True),
            ("embeddings (control)", "embed_tokens", False)]


def wmap(d):
    return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]


def tensor(d, wm, key):
    with safe_open(f"{d}/{wm[key]}", framework="pt") as f:
        return f.get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--per-surface", type=int, default=2, help="tensors sampled per surface")
    A = ap.parse_args()

    wb, wm = wmap(A.base), wmap(A.merged)
    missing = set(wb) - set(wm)
    assert not missing, f"merged checkpoint is missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}"
    print(f"[verify] {len(wm)} tensors, same key set as base", flush=True)

    bad = []
    for label, sub, must_change in SURFACES:
        keys = [k for k in wb if sub in k and "vision_tower" not in k][: A.per_surface]
        if not keys:
            print(f"[verify] {label:24s} no tensors matched {sub!r} -- SKIPPED", flush=True)
            continue
        deltas = []
        for k in keys:
            d = float((tensor(A.base, wb, k).float()
                       - tensor(A.merged, wm, k).float()).abs().max())
            deltas.append(d)
        worst = max(deltas) if must_change else min(deltas)
        ok = (worst > 0) if must_change else (max(deltas) == 0)
        flag = "OK " if ok else "FAIL"
        want = "must differ" if must_change else "must be identical"
        print(f"[verify] {flag} {label:24s} max|base-merged| = {max(deltas):.6f}  ({want})",
              flush=True)
        if not ok:
            bad.append(f"{label}: {want} but max delta {max(deltas):.6f}")

    if bad:
        print("\n[verify] MERGE VERIFICATION FAILED:", flush=True)
        for b in bad:
            print("   -", b, flush=True)
        sys.exit(1)
    print("[verify] all trained surfaces present in the merged checkpoint -- PASS", flush=True)


if __name__ == "__main__":
    main()
