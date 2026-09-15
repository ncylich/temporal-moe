#!/usr/bin/env python3
"""Rewrite a merged Qwen3.5 checkpoint as a pure text model, in place.

train_gemma_ce.py --family qwen35 --merge-out writes a config for the TEXT-ONLY class
(Qwen3_5MoeForCausalLM / model_type qwen3_5_moe_text) but leaves the weight keys under
the multimodal prefix `model.language_model.*`, and drops the vision tower and the MTP
head entirely. vLLM then instantiates Qwen3_5Model, looks for `layers.*`, and fails with
"no module or parameter named 'language_model'".

This makes the checkpoint self-consistent with the config it already declares:
  * model.language_model.<rest>  ->  model.<rest>
  * model.visual.*               ->  dropped (vision is unused in this research)
  * mtp.*                        ->  dropped (multi-token-prediction head, unused)
Expert tensors stay in the expanded per-expert layout save_pretrained produced; vLLM's
fused-MoE loader accepts that form.

    textify_qwen_merge.py /root/models/qwen35-rebuild-merged
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    A = ap.parse_args()
    d = A.model_dir

    idx_path = os.path.join(d, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]

    def rename(k):
        if k.startswith("model.visual.") or k.startswith("mtp."):
            return None
        if k.startswith("model.language_model."):
            return "model." + k[len("model.language_model."):]
        return k

    shards = sorted(set(wm.values()))
    new_map, kept, dropped = {}, 0, 0
    for shard in shards:
        p = os.path.join(d, shard)
        out = {}
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                nk = rename(k)
                if nk is None:
                    dropped += 1
                    continue
                out[nk] = f.get_tensor(k)
                new_map[nk] = shard
                kept += 1
        tmp = p + ".tmp"
        save_file(out, tmp, metadata={"format": "pt"})
        os.replace(tmp, p)
        print(f"[textify] {shard}: wrote {len(out)} tensors", flush=True)
        del out

    idx["weight_map"] = new_map
    idx["metadata"] = {"total_size": sum(os.path.getsize(os.path.join(d, s))
                                         for s in shards)}
    json.dump(idx, open(idx_path, "w"), indent=1)
    print(f"[textify] kept {kept}, dropped {dropped} (visual + mtp)", flush=True)
    print(f"[textify] {d} is now a text-only checkpoint", flush=True)


if __name__ == "__main__":
    main()
