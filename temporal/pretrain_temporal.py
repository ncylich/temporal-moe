#!/usr/bin/env python3
"""Training entrypoint for the temporal (rolling-residency) MoE.

Installs the rolling-residency router patch (temporal_router.install) and then runs Megatron's
normal GPT pretrain loop — identical to pretrain_gpt.py, only the router selection differs.
Mirrors analysis/probes/expert_load.py: invoked by run.sh (TEMPORAL=1) from inside Megatron-LM/.
"""
import os, sys

sys.path.insert(0, os.getcwd())                              # run.sh cd's to Megatron-LM/ -> import megatron
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import temporal pkg

from temporal import temporal_router
temporal_router.install()                                   # patch TopKRouter.forward before model build

if __name__ == "__main__":
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        pretrain_gpt.train_valid_test_datasets_provider,
        pretrain_gpt.model_provider,
        ModelType.encoder_or_decoder,
        pretrain_gpt.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
