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

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
