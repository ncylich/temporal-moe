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


def install():
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
                spans.append((req_id, n, computed < prompt))
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
    print("[vllm_glue] runner + olmoe/qwen3_5/gemma4 patches installed", flush=True)
