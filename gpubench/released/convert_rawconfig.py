#!/usr/bin/env python3
"""Run llama.cpp's convert_hf_to_gguf.py with the checkpoint's raw config.json instead of transformers'
AutoConfig. The converter falls back to config.json when AutoConfig fails; this wrapper makes it fail.
Needed for Gemma4 with transformers 5.23, whose Gemma4 config drops `global_head_dim`, a key the engine's
(July 2026) converter reads. Usage: convert_rawconfig.py <path/to/convert_hf_to_gguf.py> <converter args...>
"""
import runpy, sys
import transformers


def _refuse(*a, **k):
    raise RuntimeError("raw config.json requested")


transformers.AutoConfig.from_pretrained = _refuse
conv = sys.argv[1]
sys.argv = [conv] + sys.argv[2:]
sys.path.insert(0, __import__("os").path.dirname(conv))
runpy.run_path(conv, run_name="__main__")
