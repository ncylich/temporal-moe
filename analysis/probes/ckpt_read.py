#!/usr/bin/env python3
"""Read tensors out of a Megatron distributed-checkpoint (DCP) directory WITHOUT building the model.

The 1e18/1e19 runs save torch-DCP checkpoints (iter_XXXXXXX/*.distcp). There is no existing
loader in the repo for reading raw weights (router_probe.py hooks a live model instead), so this
small reader is the one piece of new plumbing the stability probes need. Reused by Part A
(stability_weights) and Part E (stability_fakequant).

The DCP metadata pickle references megatron classes, so `megatron` must be importable, and
importing megatron pulls in TransformerEngine, which needs the nvidia cudnn/cublas libs on the
loader path. Source scripts/env.sh, which derives $NV and sets CUDNN_PATH and LD_LIBRARY_PATH,
then put Megatron-LM and the repo root on PYTHONPATH:

    . scripts/env.sh
    PYTHONPATH="$ROOT/Megatron-LM:$ROOT" "$PY" analysis/probes/delex_structural.py

Everything here runs on CPU.
"""
import os, torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def iter_dir(run_ckpt_dir):
    """run_ckpt_dir/latest_checkpointed_iteration.txt -> run_ckpt_dir/iter_XXXXXXX."""
    it = open(os.path.join(run_ckpt_dir, "latest_checkpointed_iteration.txt")).read().strip()
    return os.path.join(run_ckpt_dir, f"iter_{int(it):07d}")


def weight_keys(reader):
    """All real model weight tensor keys (drop optimizer state, TE _extra_state, non-tensors)."""
    sm = reader.read_metadata().state_dict_metadata
    out = {}
    for k, m in sm.items():
        if k.startswith("optimizer") or "_extra_state" in k:
            continue
        size = getattr(m, "size", None)
        dtype = getattr(getattr(m, "properties", None), "dtype", None)
        if size is None or dtype is None:
            continue
        out[k] = (tuple(size), dtype)
    return out


def load(iter_path, keys):
    """Load the given keys from a DCP iter dir into a {key: cpu tensor} dict."""
    reader = FileSystemReader(iter_path)
    meta = weight_keys(reader)
    sd = {k: torch.empty(meta[k][0], dtype=meta[k][1]) for k in keys}
    dcp.load(sd, storage_reader=reader)
    return sd
