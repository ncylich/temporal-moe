#!/usr/bin/env python3
"""Thin wrapper to run lm-evaluation-harness's CLI after patching a transformers-5.x rename.
The vendored harness's hf_vlms model references transformers.AutoModelForVision2Seq (renamed to
AutoModelForImageTextToText in transformers 5.x) at import time; alias it before lm_eval imports,
then hand off to lm_eval's cli_evaluate (which parses sys.argv). Used by run.sh's LMEVAL branch.
"""
import sys, types
# The vendored harness eagerly imports VLM backends (hf_vlms / vllm_vlms) that reference
# transformers.AutoModelForVision2Seq, renamed in transformers 5.x. We only need the megatron_lm
# text model, so pre-stub the VLM backend modules in sys.modules to skip their import entirely
# (non-invasive: no edit to the vendored tree). Registry decorators for text models still run.
for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)
# The pod's system torchvision is built against a different torch (its _C.so fails to load), and
# `evaluate` -> transformers.pipelines -> image_utils imports it whenever transformers believes it is
# installed. The megatron_lm text model never touches vision, so declare torchvision absent.
import importlib
for _pkg, _flags in (("torchvision", ("is_torchvision_available", "is_torchvision_v2_available")),
                     ("torchaudio", ("is_torchaudio_available",))):
    try:
        importlib.import_module(_pkg)
    except Exception:
        import transformers.utils.import_utils as _iu
        import transformers.utils as _tu
        for _name in _flags:
            setattr(_iu, _name, lambda: False)
            setattr(_tu, _name, lambda: False)
        print(f"[run_lmeval] {_pkg} unusable here; marked unavailable for transformers",
              file=sys.stderr, flush=True)
print("[run_lmeval] stubbed VLM backends (hf_vlms, vllm_vlms)", file=sys.stderr, flush=True)

# TEMPORAL=1: install the rolling-residency router so temporal checkpoints are evaluated in their
# NATIVE (masked) regime — matching how t19_lmeval*.csv scored them. Without this, the megatron_lm
# model uses plain top-k routing (the UNMASKED regime), which degrades temporal models (~0.05-0.07).
import os
if os.environ.get("TEMPORAL", "0") == "1":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from temporal import temporal_router
    temporal_router.install()
    print("[run_lmeval] installed temporal residency router (native regime)", file=sys.stderr, flush=True)

# huggingface_hub >= 1.0 no longer resolves canonical dataset names ("hellaswag"), which the
# vendored task YAMLs still use. Map them to the namespaced repos the hub redirects to (same data,
# parquet-converted) at load time rather than editing the vendored tree.
import datasets as _ds
_CANON = {"hellaswag": "Rowan/hellaswag", "openbookqa": "allenai/openbookqa", "piqa": "ybisk/piqa",
          "winogrande": "allenai/winogrande", "ai2_arc": "allenai/ai2_arc", "sciq": "allenai/sciq",
          "boolq": "google/boolq", "lambada_openai": "EleutherAI/lambada_openai",
          "super_glue": "aps/super_glue"}
# piqa's namespaced repo is still script-based (datasets >= 4 refuses scripts); the hub's
# auto-converted parquet branch carries the same rows.
_PARQUET_REV = {"ybisk/piqa": "refs/convert/parquet"}
_orig_load = _ds.load_dataset
def _load(path, *a, **kw):
    path = _CANON.get(path, path)
    if path in _PARQUET_REV and "revision" not in kw:
        kw["revision"] = _PARQUET_REV[path]
    return _orig_load(path, *a, **kw)
_ds.load_dataset = _load
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
