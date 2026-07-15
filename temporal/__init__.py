"""Temporal MoE rolling-residency router (public API re-export).

The shipped core mechanism (rolling-residency routing, the Triton/CUDA-graph scan, the router
forward + install) lives in `temporal.temporal_router`. The experimental, default-off,
negative-result ablation knobs (momentum, aux-free trigger, coherence/anticipatory/bursty losses,
nomination head) live in `temporal.ablation_mechanisms` and are re-exported here for reproducibility
of results/ablations/*.csv.
"""
from .temporal_router import (
    compute_resident_mask,
    compute_resident_mask_accel,
    temporal_forward,
    banner_knobs,
    install,
)
from .ablation_mechanisms import (
    momentum_shaped_scores,
    gate_momentum_scores,
    anticipatory_target,
    anticipatory_bce_loss,
    nomination_head_logits,
    head_trigger_bonus,
    centered_demand_labels,
    head_centered_bonus,
    head_selection_active,
    bursty_window_loss,
    auxfree_trigger_scores,
    coherence_bce_loss,
)
