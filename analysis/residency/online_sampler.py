#!/usr/bin/env python3
"""In-process on-policy sampler for train_gemma_ce.py: a vLLM engine that sleeps during
training and, every N optimizer steps, wakes up, receives the CURRENT adapter's weights
merged on the GPU, samples the student under the residency rule, and goes back to sleep.

Why in-process. The trainer holds ~55 GB at rest (bf16 weights + LoRA + optimizer) and peaks
~95 GB in a step; a vLLM engine needs ~52 GB of weights plus KV. Both awake do not fit on one
H200, but vLLM's sleep mode (weights to pinned host RAM, KV freed, CUDA graphs kept) makes
the pair fit with one asleep, and in one process the weight hand-off is a GPU->GPU copy.

Weight sync. No disk, no HF merge_and_unload. Per trainable surface the merged tensor is
formed on the GPU exactly as the disk merge forms it (bf16, same op order, so the synced
engine is bit-identical to a merged checkpoint) and streamed to vLLM's load_weights under
the HF checkpoint name:
  expert LoRA   W_grouped(E,in,out) + scale*bmm(A,B) -> transpose to (E,out,in)
  attention LoRA (peft)   base.weight + get_delta_weight()
  router / norms   the trained tensors themselves
About 45 GB per sync, layer by layer (one ~1.5 GB temporary at a time).

Sampling reproduces selfgen_traj.py: card-recipe sampling (T=0.7, top-p 0.8), residency
rule on generated tokens, prefill free; returns trainer rows {ids, prompt_len}.
"""
import os
import random
import re
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                                                         # noqa: E402

_TAIL = re.compile(r"((?:layers\.\d+\.).*$)")


def _hf_name(param_or_module_name, suffix=""):
    """Trainer (peft-wrapped) name -> HF checkpoint name for gemma4's language model."""
    m = _TAIL.search(param_or_module_name)
    if m:
        return f"model.language_model.{m.group(1)}{suffix}"
    if param_or_module_name.endswith("norm.weight") or param_or_module_name.endswith("norm"):
        return "model.language_model.norm.weight" if suffix == "" or suffix == ".weight" else None
    return None


class OnlineSampler:
    def __init__(self, model, base_path, R, swaps, prompts_path, quota, max_new=1024,
                 gpu_mem=0.5, max_model_len=2560, seed=0, arch="gemma4"):
        assert arch == "gemma4", "online sampler is written for gemma4 checkpoint names"
        import vllm_glue
        import vllm_residency  # noqa: F401
        from decode_state import DEC
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        import json
        self.model, self.R, self.swaps, self.DEC = model, R, swaps, DEC
        vllm_glue.install()
        t0 = time.time()
        self.llm = LLM(model=base_path, enable_sleep_mode=True, gpu_memory_utilization=gpu_mem,
                       max_model_len=max_model_len, **vllm_glue.llm_kwargs())
        self.vmodel = self._find_model()
        self.tok = AutoTokenizer.from_pretrained(base_path)
        prompts = [json.loads(l) for l in open(prompts_path)]
        if quota:
            q = {k: int(v) for k, v in (x.split("=") for x in quota.split(","))}
            took = {k: 0 for k in q}; sel = []
            for p in prompts:
                ln = p.get("lane", "?")
                if ln in q and took[ln] < q[ln]:
                    took[ln] += 1; sel.append(p)
            prompts = sel
        self.prompts = [p.get("prompt") or p.get("text") for p in prompts]
        self.rng = random.Random(seed); self.rng.shuffle(self.prompts); self.cursor = 0
        self.sp_kw = dict(temperature=0.7, top_p=0.8, max_tokens=max_new)
        self.SamplingParams = SamplingParams
        self.n_refresh = 0
        self.llm.sleep(level=1)
        print(f"[online] engine up and asleep in {time.time()-t0:.0f}s; {len(self.prompts)} prompts "
              f"(quota {quota}); sampling R={R} swaps={swaps}", flush=True)

    def _find_model(self):
        cands = ["llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
                 "llm_engine.engine_core.model_executor.driver_worker.worker.model_runner.model",
                 "llm_engine.model_executor.driver_worker.worker.model_runner.model",
                 "llm_engine.model_executor.driver_worker.model_runner.model"]
        for path in cands:
            obj = self.llm
            try:
                for a in path.split("."):
                    obj = getattr(obj, a)
                if hasattr(obj, "load_weights"):
                    return obj
            except AttributeError:
                continue
        raise RuntimeError("[online] cannot locate the in-process vLLM model object")

    # ------------------------------------------------------------------ weights
    def _pairs(self):
        """(hf_name, merged tensor) for every trainable surface, one layer at a time."""
        mod_names = {id(m): n for n, m in self.model.named_modules()}
        with torch.no_grad():
            for mod in self.model.modules():
                if hasattr(mod, "elora_gu_A"):
                    base = _hf_name(mod_names[id(mod)])
                    s = mod.elora_scale
                    gu = mod.gate_up_proj.data + s * torch.bmm(mod.elora_gu_A.data, mod.elora_gu_B.data)
                    yield base + ".gate_up_proj", gu.transpose(1, 2).contiguous()
                    del gu
                    dp = mod.down_proj.data + s * torch.bmm(mod.elora_dp_A.data, mod.elora_dp_B.data)
                    yield base + ".down_proj", dp.transpose(1, 2).contiguous()
                    del dp
                elif hasattr(mod, "lora_A") and hasattr(mod, "base_layer") and "default" in getattr(mod, "lora_A", {}):
                    base = _hf_name(mod_names[id(mod)])
                    # peft merge() does `weight.data += delta` with the delta in the LoRA dtype (fp32 here):
                    # one rounding. Casting the delta to bf16 first double-rounds (1 ulp off, measured).
                    w = mod.base_layer.weight.data.clone()
                    w += mod.get_delta_weight("default")
                    yield base + ".weight", w
            for n, p in self.model.named_parameters():
                if p.requires_grad and "lora_" not in n and "elora_" not in n:
                    hf = _hf_name(n)
                    if hf is None or "vision" in n:
                        continue
                    yield hf, p.data

    def sync(self):
        torch.cuda.empty_cache()
        t0 = time.time(); self.llm.wake_up(); t1 = time.time()
        loaded = self.vmodel.load_weights(self._pairs())
        torch.cuda.synchronize()
        print(f"[online] wake {t1-t0:.1f}s, weight sync {time.time()-t1:.1f}s ({len(loaded)} engine params)", flush=True)
        return loaded

    # ------------------------------------------------------------------ sampling
    def sample(self, n, greedy=False, prompts=None, constrained=True, max_tokens=None):
        if prompts is None:
            prompts = []
            for _ in range(n):
                prompts.append(self.prompts[self.cursor % len(self.prompts)]); self.cursor += 1
        msgs = [[{"role": "user", "content": t}] for t in prompts]
        mt = max_tokens or self.sp_kw["max_tokens"]
        sp = self.SamplingParams(**({"temperature": 0.0, "max_tokens": mt} if greedy
                                    else dict(self.sp_kw, max_tokens=mt, seed=1234 + self.n_refresh)))
        self.DEC.update(on=constrained, R=self.R, swaps=self.swaps)
        self.DEC["state"].clear()
        t0 = time.time()
        outs = self.llm.chat(msgs, sp, use_tqdm=False)
        rows, ntok = [], 0
        for t, o in zip(prompts, outs):
            enc = self.tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                               tokenize=True, return_dict=True)
            pids = list(enc["input_ids"]); gids = list(o.outputs[0].token_ids); ntok += len(gids)
            rows.append({"ids": torch.tensor(pids + gids, dtype=torch.int32), "prompt_len": len(pids)})
        dt = time.time() - t0
        print(f"[online] sampled {len(rows)} rows, {ntok} tokens in {dt:.0f}s ({ntok/max(dt,1e-6):.0f} tok/s)", flush=True)
        return rows

    def sleep(self):
        self.llm.sleep(level=1)
        torch.cuda.empty_cache()

    def refresh(self, n):
        """sync -> sample n rows from the current adapter -> sleep. Returns trainer rows."""
        self.n_refresh += 1
        self.sync()
        rows = self.sample(n)
        self.sleep()
        return rows
