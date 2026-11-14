#!/usr/bin/env python3
"""vLLM glue for decode-time residency: two class-level patches, applied BEFORE LLM().

    runner patch   GPUModelRunner.execute_model publishes this step's spans
                   (req_id, n_tokens, is_prefill) in input-batch order to
                   vllm_residency.set_step. is_prefill = computed < prompt_len.
    model patch    each arch's MoE forward masks router_logits through
                   vllm_residency.apply (keyed by MoE-module id) before FusedMoE.

Requirements this module enforces via env (must be set before vllm engine creation):
    VLLM_ENABLE_V1_MULTIPROCESSING=0  -- engine core in-process so patches reach it
plus, at LLM() time: enforce_eager=True (stateful python hooks cannot live inside CUDA
graph replay) and enable_prefix_caching=False (cached prompt blocks are never
recomputed, so the observe phase would miss them and decode would start near-cold --
manifests as gibberish under tight R while the HF path stays coherent). Free arms run
the SAME patched path with DEC.on=False (exact no-op).

OLMoE note: vLLM builds FusedMoE with renormalize=False -- the model's own preserve
convention -- so -inf masking before the fused kernel is semantically identical to the
audited HF path.
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import sys                                                            # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_residency as VR                                           # noqa: E402


def llm_kwargs():
    """Engine kwargs for the residency stack. Fast walker (default): full CUDA graphs on
    decode-only steps, no torch.compile (CompilationMode.NONE), so the fused kernel is
    captured and replayed and prefill/mixed steps stay eager where the python part of the
    walker runs. TEMPORAL_EAGER=1 or a non-fast TEMPORAL_WALKER keeps the old
    enforce_eager=True behaviour. Prefix caching stays off in every mode (see module doc)."""
    # TEMPORAL_MAMBA_FP32=1: keep the linear-attention (gated delta net) state cache in fp32 during
    # decode; hybrid models default to the model dtype (bf16) and drift from HF over long decodes.
    extra = {"mamba_ssm_cache_dtype": "float32"} if os.environ.get("TEMPORAL_MAMBA_FP32") == "1" else {}
    if os.environ.get("TEMPORAL_FAST_PP", "1") == "1":       # presence penalty via a persistent mask (fast_penalty.py); native halves throughput
        from fast_penalty import FastPresencePenalty
        extra["logits_processors"] = [FastPresencePenalty]
    if os.environ.get("TEMPORAL_EAGER") == "1" or VR._WALKER != "fast":
        return {"enforce_eager": True, "enable_prefix_caching": False, **extra}
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
    return {"enforce_eager": False, "enable_prefix_caching": False, **extra,
            "compilation_config": CompilationConfig(mode=CompilationMode.NONE,
                                                    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)}


def _install_fast_penalty():
    """TEMPORAL_FAST_PP=1 (default): any SamplingParams with presence_penalty > 0 handed to LLM.generate/chat
    is rewritten to the FastPresencePenalty logits processor (fast_penalty.py; same math, 1.8x the
    throughput at batch 256). Transparent to lm-eval and every producer."""
    if os.environ.get("TEMPORAL_FAST_PP", "1") != "1":
        return
    from vllm import LLM
    from fast_penalty import KEY
    if getattr(LLM, "_tmoe_fast_pp", False):
        return
    def rewrite(sp):
        if sp is None or isinstance(sp, (list, tuple)):
            return [rewrite(x) for x in sp] if sp is not None else sp
        pp = getattr(sp, "presence_penalty", 0.0) or 0.0
        if pp <= 0:
            return sp
        sp = sp.clone(); sp.presence_penalty = 0.0
        sp.extra_args = dict(sp.extra_args or {}, **{KEY: pp})
        return sp
    for name in ("generate", "chat"):
        orig = getattr(LLM, name)
        def wrapped(self, *a, _orig=orig, **k):
            if "sampling_params" in k:
                k["sampling_params"] = rewrite(k["sampling_params"])
            elif len(a) >= 2:
                a = (a[0], rewrite(a[1])) + tuple(a[2:])
            return _orig(self, *a, **k)
        setattr(LLM, name, wrapped)
    LLM._tmoe_fast_pp = True


def install():
    _install_fast_penalty()
    if VR._WALKER in ("fast", "cache_bias"):   # the evals read swap traffic from the router module
        from temporal import temporal_router as TR
        TR.swap_stats = VR.swap_stats
    # transformers 5.15 heterogeneity guard: gemma4 marks head_dim per-layer and vLLM
    # reads it globally; hf_overrides does not reach every config object vLLM constructs,
    # so permit global access at the mixin class level. gemma4-26B is homogeneous in
    # practice; the free-arm benchmark scores are the sanity check.
    try:
        from transformers.integrations.heterogeneity.configuration_utils import (
            HeterogeneousConfigMixin,
        )
        HeterogeneousConfigMixin.allow_global_per_layer_attribute_access = True
    except ImportError:
        pass

    # Hook AFTER _update_states: execute_model first syncs input_batch/requests with the
    # scheduler output (new requests added, finished removed), and only then is the batch
    # the authority on this step's token-stream order.
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    orig_update = GPUModelRunner._update_states

    def _update_states(self, scheduler_output, *a, **k):
        ret = orig_update(self, scheduler_output, *a, **k)
        nst = getattr(scheduler_output, "num_scheduled_tokens", None)
        spans = None
        if nst:
            spans = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                n = nst.get(req_id)
                if not n:      # row not scheduled this step (e.g. just finished): 0 tokens
                    continue
                computed = int(self.input_batch.num_computed_tokens_cpu[i])
                prompt = int(self.requests[req_id].num_prompt_tokens)
                # RESUME: when a submitted "prompt" is original-prompt + previously
                # generated tokens, the generated part must be walked under the rule
                # (it was generated under it), while the original prompt stays free.
                # Keyed off the prompt token IDs themselves, so it does not depend on
                # how vLLM assigns request ids.
                if _DEC.get("resume_map") and req_id not in _DEC["enforce_from"]:
                    ids = tuple(self.requests[req_id].prompt_token_ids)
                    key = (len(ids), hash(ids[:16]), hash(ids[-16:]))
                    if key in _DEC["resume_map"]:
                        _DEC["enforce_from"][req_id] = _DEC["resume_map"][key]
                spans.append((req_id, n, computed < prompt, computed))
        VR.set_step(spans)
        return ret

    GPUModelRunner._update_states = _update_states

    from vllm.model_executor.models import olmoe as mo

    def olmoe_moe_forward(self, hidden_states):
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits, _ = self.gate(hidden_states)
        masked = VR.apply(id(self), router_logits)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=masked
        )
        if masked is not router_logits:
            # GATE-MASS PRESERVE (the historical OLMoE bug, vLLM edition): the fused
            # kernel softmaxes the MASKED row, so the k selected residents share mass
            # 1.0 -- the ~2.5x renorm inflation. OLMoE is norm_topk_prob=False: correct
            # weights are the FULL softmax masses at the selected experts. Both are the
            # same weights up to a per-token scalar, so scale the output by the full
            # softmax mass of the selection. Rows left unmasked (prefill) scale by 1.
            import torch
            import torch.nn.functional as F
            k = self.experts.moe_config.experts_per_token
            full = F.softmax(router_logits.float(), dim=-1)
            sel = masked.float().topk(k, dim=-1).indices          # the kernel's selection
            mass = full.gather(-1, sel).sum(-1)
            mass = torch.where(torch.isinf(masked).any(-1), mass,
                               torch.ones_like(mass))
            final_hidden_states = final_hidden_states * mass.to(
                final_hidden_states.dtype).unsqueeze(-1)
        return final_hidden_states.view(orig_shape)

    mo.OlmoeMoE.forward = olmoe_moe_forward

    # qwen3.5 (MoE block shared with qwen3-next): the model's own convention
    # renormalizes over the selection, so masking logits is exact -- no mass
    # correction. Shared expert lives inside FusedMoE and stays free (correct:
    # not a swap candidate). The copied body is the external-router, non-SP path;
    # constrained runs on other paths fail loudly instead of silently free-running.
    from vllm.model_executor.models import qwen3_next as mq
    from decode_state import DEC as _DEC

    # Force EXTERNAL routing: the engine may fuse the gate into the MoE runner
    # (is_internal_router), which hides router logits from interception. Dropping the
    # runner's gate reference before the first forward selects the reference external
    # path; the block still owns its gate module and computes logits itself.
    orig_q3n_init = mq.Qwen3NextSparseMoeBlock.__init__

    def q3n_init(self, *a, **k):
        orig_q3n_init(self, *a, **k)
        if getattr(self.experts, "gate", None) is not None:
            self.experts.gate = None

    mq.Qwen3NextSparseMoeBlock.__init__ = q3n_init
    orig_q3n = mq.Qwen3NextSparseMoeBlock.forward

    def q3n_forward(self, hidden_states, already_sequence_parallel=False):
        if self.experts.is_internal_router or self.is_sequence_parallel:
            assert not _DEC["on"], \
                "constrained run needs external router and no sequence parallelism"
            return orig_q3n(self, hidden_states, already_sequence_parallel)
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        router_logits, _ = self.gate(hidden_states)
        router_logits = VR.apply(id(self), router_logits)
        final = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return final.view(orig_shape)

    mq.Qwen3NextSparseMoeBlock.forward = q3n_forward

    # gemma4: custom routing function renormalizes over the selection -> masking
    # the (external) router logits is exact here too.
    from vllm.model_executor.models import gemma4 as mg
    orig_g4 = mg.Gemma4MoE.forward

    def g4_forward(self, x, router_logits):
        return orig_g4(self, x, VR.apply(id(self), router_logits))

    mg.Gemma4MoE.forward = g4_forward

    # transformers 5.15 moves gemma4's head_dim/global_head_dim into its per-layer spec
    # (deleting the plain global_head_dim the checkpoint json carries), so vLLM's
    # `getattr(config, "global_head_dim", config.head_dim)` silently falls back and
    # builds 512-dim full-attention layers at 256 -> checkpoint shape mismatch. Restore
    # the two plain attributes from the per-layer spec before any layer is constructed.
    orig_g4layer_init = mg.Gemma4DecoderLayer.__init__

    def g4layer_init(self, config, *a, **k):
        try:
            plc = config.per_layer_config
            dims = {t: getattr(plc[i], "head_dim")
                    for i, t in enumerate(config.layer_types)}
            kvs = {t: getattr(plc[i], "num_key_value_heads")
                   for i, t in enumerate(config.layer_types)}
            if "sliding_attention" in dims:
                config.head_dim = dims["sliding_attention"]
                config.num_key_value_heads = kvs["sliding_attention"]
            if "full_attention" in dims:
                config.global_head_dim = dims["full_attention"]
                config.num_global_key_value_heads = kvs["full_attention"]
        except (AttributeError, IndexError, TypeError):
            pass                          # non-heterogeneous config: nothing to restore
        orig_g4layer_init(self, config, *a, **k)

    mg.Gemma4DecoderLayer.__init__ = g4layer_init

    # LFM2.5: selection = sigmoid(logits) + expert bias, fused inside vLLM's grouped-topk
    # router. -inf logit masking is NOT safe here (sigmoid(-inf)=0, but a non-resident
    # expert's bias alone can still win selection), so reroute through a custom routing
    # function replicating the audited HF path (granularity_ladder.patch_lfm) exactly:
    # scan and mask operate on the CHOICE signal sigmoid+bias; gate weights are the raw
    # sigmoid at the selected experts, renormalized (+1e-6) and scaled. FusedMoE is a
    # FACTORY FUNCTION in this vLLM version (not a class -- rebinding its __init__ is a
    # silent no-op, which produced arm-identical generations on the first attempt), so
    # wrap the name in the lfm2_moe module namespace; the guard matches only LFM's
    # signature (sigmoid scoring + grouped topk).
    import torch
    from vllm.model_executor.models import lfm2_moe as mlf
    # vLLM renamed the factory FusedMoE -> FusedMoEFactory (0.27.x). Same signature,
    # same call site inside Lfm2MoeSparseMoeBlock, so bind whichever name this version
    # exposes -- the wrap must land on the name lfm2_moe itself calls.
    _fm_name = "FusedMoE" if hasattr(mlf, "FusedMoE") else "FusedMoEFactory"
    orig_fm = getattr(mlf, _fm_name)

    def fm_wrap(*a, **kw):
        if kw.get("scoring_func") == "sigmoid" and kw.get("use_grouped_topk"):
            bias = kw.get("e_score_correction_bias")
            renorm = kw.get("renormalize", True)
            scale = kw.get("routed_scaling_factor", 1.0)
            layer_key = object()                      # unique hashable key per MoE layer

            def route(hidden_states, gating_output, topk, renormalize):
                scores = gating_output.sigmoid()
                sel_signal = scores + bias if bias is not None else scores
                masked = VR.apply(layer_key, sel_signal)
                _, selected = torch.topk(masked, k=topk, dim=-1)
                weights = torch.gather(scores, 1, selected)
                if renorm:
                    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
                return weights * scale, selected

            kw = dict(kw, use_grouped_topk=False, num_expert_group=None,
                      topk_group=None, custom_routing_function=route)
        return orig_fm(*a, **kw)

    setattr(mlf, _fm_name, fm_wrap)

    # gpt-oss: external router logits, softmax over the selection (renormalize=True)
    # -> -inf masking before the fused kernel is exact, same as the audited HF MXFP4
    # port. Copy of the plain CUDA, non-sequence-parallel forward with the mask added.
    from vllm.model_executor.models import gpt_oss as mgo
    from vllm.platforms import current_platform
    orig_goss = mgo.MLPBlock.forward

    def goss_forward(self, x):
        if current_platform.is_rocm() or self.is_sequence_parallel:
            assert not _DEC["on"], "constrained gpt-oss needs the plain CUDA path"
            return orig_goss(self, x)
        g = self.router(x)
        g = VR.apply(id(self), g)
        return self.experts(hidden_states=x, router_logits=g)[:, : self.hidden_size]

    mgo.MLPBlock.forward = goss_forward
    print("[vllm_glue] runner + olmoe/qwen3_5/gemma4/lfm25/gpt_oss patches installed",
          flush=True)
