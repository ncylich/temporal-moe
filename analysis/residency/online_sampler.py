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
                 gpu_mem=0.5, max_model_len=2560, seed=0, arch="gemma4", temperature=0.7, top_p=0.8,
                 offload_layers=0, presence_penalty=0.0, think=False):
        assert arch in ("gemma4", "qwen35"), arch
        self.arch = arch
        # qwen35: trainer (70 GB) + engine weights (66 GB) exceed the 140 GB GPU. While the engine is
        # awake, the frozen expert base weights of the first `offload_layers` layers live on the host
        # (1.6 GB/layer); they come back before training resumes. The merge reads them through .to(cuda).
        self.offload_layers = offload_layers
        self.model = model
        self.dev = next(model.parameters()).device
        self._offload()
        import vllm_glue
        import vllm_residency  # noqa: F401
        from decode_state import DEC
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        import json
        self.model, self.R, self.swaps, self.DEC = model, R, swaps, DEC
        vllm_glue.install()
        t0 = time.time()
        # max_num_seqs 256: we never sample more rows than that at once, and hybrid (linear-attention)
        # models need one state block per decode sequence; at a 0.55 memory share only ~285 blocks
        # exist, and CUDA-graph capture refuses the 1024 default.
        self.llm = LLM(model=base_path, enable_sleep_mode=True, gpu_memory_utilization=gpu_mem,
                       max_model_len=max_model_len, max_num_seqs=256, **vllm_glue.llm_kwargs())
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
        self.sp_kw = dict(temperature=temperature, top_p=top_p, max_tokens=max_new)
        self.max_new = max_new
        # qwen's card recipe adds presence_penalty 1.5 for non-thinking chat; vLLM implements penalties with a
        # per-step vocab-sized count per sequence and it HALVES batch-256 throughput (4171 -> 2220 tok/s,
        # 2026-08-29). Off by default in the sampler (the objective and the cap bound repetition); evals keep it.
        if presence_penalty:                       # vllm_glue routes it to the fast processor (TEMPORAL_FAST_PP=1)
            self.sp_kw["presence_penalty"] = presence_penalty
        # thinking off is the default on both templates (gemma's default already emits an empty thought channel);
        # think=True opens it on both (the on-policy rows then contain the thinking tokens the teacher scores too)
        self.chat_kw = {"chat_template_kwargs": {"enable_thinking": bool(think)}}
        self.SamplingParams = SamplingParams
        self.n_refresh = 0
        self.llm.sleep(level=1)
        self._restore()
        print(f"[online] engine up and asleep in {time.time()-t0:.0f}s; {len(self.prompts)} prompts "
              f"(quota {quota}); sampling R={R} swaps={swaps}", flush=True)

    def _expert_mods(self):
        return [m for m in self.model.modules() if hasattr(m, "elora_gu_A")]

    def _offload(self):
        """Move the frozen expert base weights of the first `offload_layers` layers to pinned host
        buffers (allocated once; pinned copies run at PCIe speed, ~4x faster than pageable)."""
        if not self.offload_layers:
            return
        t0 = time.time()
        if not hasattr(self, "_pinned"):
            self._pinned = {}
        for i, m in enumerate(self._expert_mods()[:self.offload_layers]):
            for a in ("gate_up_proj", "down_proj"):
                t = getattr(m, a)
                if not t.data.is_cuda:
                    continue
                key = (i, a)
                if key not in self._pinned:
                    self._pinned[key] = torch.empty_like(t.data, device="cpu", pin_memory=True)
                self._pinned[key].copy_(t.data, non_blocking=True)
                t.data = self._pinned[key]
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        print(f"[online] offloaded {self.offload_layers} layers of expert base weights to pinned host memory in {time.time()-t0:.1f}s", flush=True)

    def _restore(self):
        if not self.offload_layers:
            return
        t0 = time.time()
        for m in self._expert_mods():
            for a in ("gate_up_proj", "down_proj"):
                t = getattr(m, a)
                if not t.data.is_cuda:
                    t.data = t.data.to(self.dev, non_blocking=True)
        torch.cuda.synchronize()
        print(f"[online] restored expert base weights to the GPU in {time.time()-t0:.1f}s", flush=True)

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
    def _name(self, n, suffix=""):
        return _hf_name(n, suffix)              # raw-base engine: model.language_model.* for both families

    def _pairs(self):
        """(hf_name, merged tensor) for every trainable surface, one layer at a time."""
        mod_names = {id(m): n for n, m in self.model.named_modules()}
        with torch.no_grad():
            for mod in self.model.modules():
                if hasattr(mod, "elora_gu_A"):
                    base = self._name(mod_names[id(mod)])
                    s = mod.elora_scale
                    dev = mod.elora_gu_A.device
                    gu = mod.gate_up_proj.data.to(dev) + s * torch.bmm(mod.elora_gu_A.data, mod.elora_gu_B.data)   # (E,H,2I)
                    dp = mod.down_proj.data.to(dev) + s * torch.bmm(mod.elora_dp_A.data, mod.elora_dp_B.data)      # (E,I,H)
                    yield base + ".gate_up_proj", gu.transpose(1, 2).contiguous()
                    yield base + ".down_proj", dp.transpose(1, 2).contiguous()
                    del gu, dp
                elif hasattr(mod, "lora_A") and hasattr(mod, "base_layer") and "default" in getattr(mod, "lora_A", {}):
                    base = self._name(mod_names[id(mod)])
                    # peft merge() does `weight.data += delta` with the delta in the LoRA dtype (fp32 here):
                    # one rounding. Casting the delta to bf16 first double-rounds (1 ulp off, measured).
                    w = mod.base_layer.weight.data.clone()
                    w += mod.get_delta_weight("default")
                    yield base + ".weight", w
            for n, p in self.model.named_parameters():
                if p.requires_grad and "lora_" not in n and "elora_" not in n:
                    hf = self._name(n)
                    if hf is None or "vision" in n or "visual" in n or "mtp" in n:
                        continue
                    yield hf, p.data

    def sync(self):
        self._offload()
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
        outs = self.llm.chat(msgs, sp, use_tqdm=False, **self.chat_kw)
        rows, ntok = [], 0
        for t, o in zip(prompts, outs):
            enc = self.tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                               tokenize=True, return_dict=True)
            pids = list(enc["input_ids"]); gids = list(o.outputs[0].token_ids); ntok += len(gids)
            rows.append({"ids": torch.tensor(pids + gids, dtype=torch.int32), "prompt_len": len(pids)})
        dt = time.time() - t0
        print(f"[online] sampled {len(rows)} rows, {ntok} tokens in {dt:.0f}s ({ntok/max(dt,1e-6):.0f} tok/s)", flush=True)
        if not greedy:                                    # what the student is producing right now
            cap = sum(len(o.outputs[0].token_ids) >= mt for o in outs)
            txt = [o.outputs[0].text for o in outs]; nch = sum(len(t_) for t_ in txt) or 1
            dig = sum(c.isdigit() for t_ in txt for c in t_) / nch; eq = sum(t_.count("=") for t_ in txt) / len(txt)
            print(f"[online] sample stats: mean len {ntok/len(rows):.0f} tok, cap-hit {100*cap/len(rows):.0f}%, "
                  f"digit chars {100*dig:.1f}%, '=' per row {eq:.1f}", flush=True)
        return rows

    def score(self, prompts, gens, constrained=False):
        """Per-token logprobs of given generations under the current engine (teacher-forced via prompt_logprobs)."""
        self.DEC.update(on=constrained, R=self.R, swaps=self.swaps); self.DEC["state"].clear()
        sp = self.SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        ids, plens = [], []
        for t, g in zip(prompts, gens):
            enc = self.tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                               tokenize=True, return_dict=True)
            p_ = list(enc["input_ids"]); ids.append(p_ + list(g)); plens.append(len(p_))
        outs = self.llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
        return [[o.prompt_logprobs[j][x[j]].logprob for j in range(pl, len(x))] for o, x, pl in zip(outs, ids, plens)]

    def sleep(self):
        self.llm.sleep(level=1)
        torch.cuda.empty_cache()
        self._restore()

    def refresh(self, n):
        """sync -> sample n rows from the current adapter -> sleep. Returns trainer rows."""
        self.n_refresh += 1
        self.sync()
        rows = self.sample(n)
        self.sleep()
        return rows
