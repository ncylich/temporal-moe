#!/usr/bin/env python3
"""Run llama.cpp's convert_hf_to_gguf.py with the checkpoint's raw config.json instead of transformers'
AutoConfig for the model hparams. The converter falls back to config.json when AutoConfig fails; this
wrapper makes that one call fail (the tokenizer still uses AutoConfig).
Needed for Gemma4 with transformers 5.23, whose Gemma4 config drops `global_head_dim`, a key the engine's
(July 2026) converter reads. Usage: convert_rawconfig.py <path/to/convert_hf_to_gguf.py> <converter args...>
"""
import inspect, runpy, sys
import transformers

_orig = transformers.AutoConfig.from_pretrained


def _refuse_in_load_hparams(*a, **k):
    # only the converter's hparams loading is redirected; tokenizer loading keeps AutoConfig
    if any(fr.function == "load_hparams" for fr in inspect.stack()[1:4]):
        raise RuntimeError("raw config.json requested")
    return _orig(*a, **k)


transformers.AutoConfig.from_pretrained = _refuse_in_load_hparams
conv = sys.argv[1]
sys.argv = [conv] + sys.argv[2:]
sys.path.insert(0, __import__("os").path.dirname(conv))
runpy.run_path(conv, run_name="__main__")
