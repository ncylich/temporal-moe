#!/usr/bin/env python3
"""Per-Layer Embeddings for OLMoE residency adaptation (PLE_PLAN.md §2).

A token-indexed lookup added to each decoder layer's OUTPUT:

    h   = x + Attn(LN1(x))
    out = h + MoE(LN2(h)) + g_l * PLE[tok, l]

Placement is post-MoE ONLY. The pre-MoE variant is explicitly out of scope (PLE_PLAN.md §13):
feeding PLE into the MoE input would change routing decisions and corrupt the trajectory the norm
gains adapted to. Nothing here can produce it; there is no flag for it.

Two forms, selected by `rank`:

  factored (rank = 32 / 128 / 512)
      U [vocab, r]           per-token code, flash-resident, the only token-indexed tensor
      V [r, layers, hidden]  shared basis, RAM-resident
      One code per token receives gradient from all layers.

  full (rank = "full")
      P [vocab, layers, hidden], unfactored. NOT r = layers*hidden: at that rank the factorization
      would store both U and V and be larger than the table it replaces.

INITIALIZATION, and why it is not what §2 literally says
--------------------------------------------------------
§2 asks for a zero-initialized table AND a gate "initialized so the branch starts inert". Doing
both at once produces a branch that can never train: with U = 0 the contribution is already
identically zero, and a zero gate additionally kills the gradient reaching U, so dL/dU = dL/dV =
dL/dg = 0 and every tensor is a fixed point. The branch would stay dead for the whole run and the
cell would report exactly the C-recipe number.

What is implemented instead, which satisfies every property §2 actually wants:

    U (or P) zero-init   -> contribution is exactly zero at step 0, so the model is bit-identical
                            to the C recipe, and parity is free
    V random-init        -> dL/dU = g * (upstream @ V^T) is nonzero, so U can move
    g init 1.0           -> learned per layer, but starts at unity rather than zero

The gate is redundant for inertness (the zero table already provides it) and is kept only because
§2 specifies a learned per-layer scale.

THE ZERO PROPERTY (PLE_PLAN.md §4 item 5) then holds by construction and is preserved through
training, not merely at init:
  - embedding gradients are sparse, so a row whose token never appears receives no loss gradient;
  - AdamW decoupled weight decay multiplies a row by (1 - lr*wd), and 0 * anything is 0;
  - Adam's own update on a never-touched row is m_hat/(sqrt(v_hat)+eps) = 0/eps = 0.
So uncovered rows stay bit-zero, and a forward pass on an uncovered token adds exactly 0.0 —
matching the no-PLE model bit-for-bit, not approximately. zero_check.py tests this rather than
assuming it.

The table is held in fp32 and the looked-up rows are cast to the model dtype at use. Keeping it
fp32 removes the need for the separate fp32 master copy the other trainers carry, which at full
rank saves 6.6 GB rather than costing it.
"""

import torch
import torch.nn as nn

RANKS = (32, 128, 512, "full")


class FactoredPLE(nn.Module):
    """The PLE table. `rank` is an int for the factored form or the string "full"."""

    def __init__(self, vocab, layers, hidden, rank, device="cuda", seed=1234):
        super().__init__()
        self.vocab, self.layers, self.hidden, self.rank = vocab, layers, hidden, rank
        self.full = (rank == "full")
        gen = torch.Generator(device="cpu").manual_seed(seed)  # own generator: never perturbs data order
        if self.full:
            # unfactored table, zero-init. No basis to initialize.
            self.P = nn.Parameter(torch.zeros(vocab, layers, hidden, dtype=torch.float32, device=device))
        else:
            self.U = nn.Parameter(torch.zeros(vocab, rank, dtype=torch.float32, device=device))
            # std = 1/sqrt(r) so that ||U @ V|| tracks ||U|| once U leaves zero.
            v = torch.empty(rank, layers, hidden, dtype=torch.float32).normal_(0.0, rank ** -0.5, generator=gen)
            self.V = nn.Parameter(v.to(device))
        self.g = nn.Parameter(torch.ones(layers, dtype=torch.float32, device=device))
        # Deliberately held-out token ids for the zero-property check (§4 item 5). Coverage is
        # ~100%, so the naturally-uncovered rows are unused vocab padding slots -- testing only
        # those would prove almost nothing, because they are rows the model had no chance to touch
        # for trivial reasons. A held-out set is rows that WERE eligible and still must be zero.
        # Zeroing the contribution in the forward (rather than masking the gradient afterwards) is
        # what makes it airtight: an unread row cannot receive gradient at all, so the row stays
        # bit-zero by the same mechanism as a genuinely absent token.
        self.register_buffer("heldout", torch.zeros(vocab, dtype=torch.bool, device=device))

    def set_heldout(self, ids):
        self.heldout.zero_()
        if ids is not None and len(ids):
            self.heldout[torch.as_tensor(ids, dtype=torch.long, device=self.heldout.device)] = True

    def table_params(self):
        """The token-indexed table: the tensors weight decay must reach (§2)."""
        return [self.P] if self.full else [self.U]

    def basis_params(self):
        """Everything else: the shared basis and the per-layer gates."""
        return [self.g] if self.full else [self.V, self.g]

    def forward(self, token_ids, layer_idx, dtype):
        """token_ids [B, S] -> [B, S, hidden] in `dtype`."""
        if self.full:
            d = self.P[token_ids, layer_idx]                     # [B,S,H]
        else:
            d = self.U[token_ids] @ self.V[:, layer_idx, :]      # [B,S,r] @ [r,H]
        if bool(self.heldout.any()):
            d = d * (~self.heldout[token_ids]).unsqueeze(-1)
        return (self.g[layer_idx] * d).to(dtype)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def fetch_bytes_per_token(self, bytes_per_elem=2):
        """Flash traffic for one token: the token-indexed row only. The basis is RAM-resident."""
        return (self.layers * self.hidden if self.full else self.rank) * bytes_per_elem

    def basis_bytes(self, bytes_per_elem=2):
        return 0 if self.full else self.rank * self.layers * self.hidden * bytes_per_elem


# ---------------------------------------------------------------- install / uninstall

_STATE = {"ple": None, "ids": None, "orig_layer_fwd": None, "orig_model_fwd": None}


def _layer_forward(self, hidden_states, *a, **kw):
    out = _STATE["orig_layer_fwd"](self, hidden_states, *a, **kw)
    ple, ids = _STATE["ple"], _STATE["ids"]
    if ple is None or ids is None:
        return out
    # post-MoE: the decoder layer has already done `residual + mlp(...)`, so `out` is the layer
    # output and adding here is exactly `out + g_l * PLE[tok, l]`. The MoE input is untouched,
    # so routing decisions are bit-identical to the no-PLE model.
    tensor = out[0] if isinstance(out, tuple) else out
    add = ple(ids, self._ple_layer_idx, tensor.dtype)
    tensor = tensor + add
    return (tensor,) + out[1:] if isinstance(out, tuple) else tensor


def _model_forward(self, input_ids=None, *a, **kw):
    # Stash the ids for the layers. inputs_embeds-only calls leave them None and the branch is
    # skipped rather than guessing, because there is no token index to look up.
    #
    # The ids are deliberately NOT restored when this returns. Gradient checkpointing (which the
    # C recipe enables) recomputes each layer's forward during backward, long after this function
    # has returned; a save/restore around the call leaves them None at recompute time, so the PLE
    # add would be silently absent from the recomputed graph and the table would receive no
    # gradient at all -- a dead branch that still logs a healthy loss curve. Verified: under
    # torch.utils.checkpoint the layer sees the ids on the forward pass and None on the recompute.
    #
    # Consequence: exactly one model forward may be in flight at a time, which holds for the C
    # recipe. A teacher/student arm that interleaves two forwards before backward (the adaptation
    # program's distillation arms D and G) would need the ids carried per-graph instead.
    _STATE["ids"] = input_ids
    return _STATE["orig_model_fwd"](self, input_ids, *a, **kw)


def install(model, rank, device="cuda", seed=1234):
    """Attach a PLE table to `model` and patch the decoder layer + model forwards. Returns it."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer, OlmoeModel

    cfg = model.config
    ple = FactoredPLE(cfg.vocab_size, cfg.num_hidden_layers, cfg.hidden_size, rank,
                      device=device, seed=seed)
    for i, layer in enumerate(model.model.layers):
        layer._ple_layer_idx = i
    if _STATE["orig_layer_fwd"] is None:
        _STATE["orig_layer_fwd"] = OlmoeDecoderLayer.forward
        _STATE["orig_model_fwd"] = OlmoeModel.forward
        OlmoeDecoderLayer.forward = _layer_forward
        OlmoeModel.forward = _model_forward
    _STATE["ple"] = ple
    return ple


def uninstall():
    """Detach the table. The patched forwards stay in place but become pass-throughs."""
    _STATE["ple"] = None


def is_installed():
    return _STATE["ple"] is not None
