#!/usr/bin/env python3
"""Fused Triton kernels for the serving-side residency walker (vllm_residency.py).

Why. The constrained vLLM path ran 2.6-3.2x slower than the free arm on every eval and
~330 tok/s on an H200 during sampling (2026-08-28). Three causes, all launch overhead:
  1. prefill observation stepped the resident state one token at a time in python
     (~8 kernel launches per prompt token per layer: a 300-token prompt is ~72k launches
     before its first generated token);
  2. the decode hot path was ~8 launches per layer per step (two index_selects, a graph
     replay plus clone, index_copy_, masked_fill, a scatter, and a full [N,E] clone);
  3. the whole model ran with enforce_eager=True because the hook branched in python.

Here the rule is two kernels. `decode_step` advances every decode row of a step in ONE
launch per layer, reading and writing a persistent per-layer state bank through index
buffers that live on the device, so it can be captured into a CUDA graph and replayed
with no python. `chunk_scan` walks one prefill/replay chunk in one launch (the sequential
recurrence in-kernel, as temporal_router._scan_kernel does for training).

Semantics are the published min_logit rule with the optional deadband RHO
(temporal_router._minlogit_step): nominee = argmax over non-resident logits, worst =
min over resident logits, swap iff nominee > worst + RHO, evict the worst. Ties break to
the first index, matching torch. All comparisons in fp32. `reference_step` /
`reference_scan` below are the plain-torch versions the tests compare against
(test_residency_kernels.py), and the walker's one-time gate compares the kernels to
them at first use and raises on any mismatch.

Swap accounting is on-device (`count[0]` swaps performed, `count[1]` decode rows
stepped) and is read once by swap_stats(), so counting costs no per-step sync. Only
DECODE tokens (including preemption replays, which are generated tokens) are counted:
prefill is observed free and its state updates are not weight movements.
"""
import torch
import triton
import triton.language as tl

NEG = float("-inf")
POS = float("inf")


@triton.jit
def _decode_kernel(logits_ptr, out_ptr, bank_ptr, rows_ptr, slots_ptr, ndec_ptr, count_ptr,
                   E, RHO, BLOCK: tl.constexpr, NSWAPS: tl.constexpr):
    d = tl.program_id(0)
    ndec = tl.load(ndec_ptr)
    live = d < ndec
    e = tl.arange(0, BLOCK)
    valid = e < E
    row = tl.load(rows_ptr + d, mask=live, other=0).to(tl.int64)
    slot = tl.load(slots_ptr + d, mask=live, other=0).to(tl.int64)
    lt = tl.load(logits_ptr + row * E + e, mask=valid & live, other=0.0).to(tl.float32)
    resident = tl.load(bank_ptr + slot * E + e, mask=valid & live, other=0) != 0
    nsw = tl.zeros((), dtype=tl.int32)
    NEGV = float("-inf")
    POSV = float("inf")
    for _sw in tl.static_range(NSWAPS):
        masked_nom = tl.where(valid & (resident == 0), lt, NEGV)
        nom_val = tl.max(masked_nom, 0)
        nom_i = tl.min(tl.where(masked_nom == nom_val, e, BLOCK), 0)
        masked_ev = tl.where(resident, lt, POSV)
        worst_val = tl.min(masked_ev, 0)
        evict_i = tl.min(tl.where(masked_ev == worst_val, e, BLOCK), 0)
        do_swap = nom_val > worst_val + RHO
        is_evict = (e == evict_i) & do_swap
        is_nom = (e == nom_i) & do_swap
        resident = (resident & (is_evict == 0)) | is_nom
        nsw += do_swap.to(tl.int32)
    tl.store(bank_ptr + slot * E + e, resident.to(tl.int8), mask=valid & live)
    outv = tl.where(resident, lt, NEGV).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * E + e, outv, mask=valid & live)
    tl.atomic_add(count_ptr, nsw.to(tl.int64), mask=live)
    tl.atomic_add(count_ptr + 1, 1, mask=live)


@triton.jit
def _chunk_scan_kernel(logits_ptr, res0_ptr, out_ptr, count_ptr, S, T0, E, RHO,
                       BLOCK: tl.constexpr, NSWAPS: tl.constexpr):
    e = tl.arange(0, BLOCK)
    valid = e < E
    resident = tl.load(res0_ptr + e, mask=valid, other=0) != 0
    NEGV = float("-inf")
    POSV = float("inf")
    if T0 == 1:                                   # unseeded: out[0] is the cold fill
        tl.store(out_ptr + e, resident.to(tl.int8), mask=valid)
    nsw = tl.zeros((), dtype=tl.int32)
    for t in range(T0, S):
        lt = tl.load(logits_ptr + t * E + e, mask=valid, other=0.0).to(tl.float32)
        for _sw in tl.static_range(NSWAPS):
            masked_nom = tl.where(valid & (resident == 0), lt, NEGV)
            nom_val = tl.max(masked_nom, 0)
            nom_i = tl.min(tl.where(masked_nom == nom_val, e, BLOCK), 0)
            masked_ev = tl.where(resident, lt, POSV)
            worst_val = tl.min(masked_ev, 0)
            evict_i = tl.min(tl.where(masked_ev == worst_val, e, BLOCK), 0)
            do_swap = nom_val > worst_val + RHO
            is_evict = (e == evict_i) & do_swap
            is_nom = (e == nom_i) & do_swap
            resident = (resident & (is_evict == 0)) | is_nom
            nsw += do_swap.to(tl.int32)
        tl.store(out_ptr + t * E + e, resident.to(tl.int8), mask=valid)
    tl.atomic_add(count_ptr, nsw.to(tl.int64))
    tl.atomic_add(count_ptr + 1, S - T0)


def _block(E):
    return max(16, 1 << (E - 1).bit_length())


def decode_step(out, logits, bank, rows, slots, ndec, count, rho, swaps, cap):
    """Advance every live decode row in place: bank[slots[d]] steps on logits[rows[d]] and
    out[rows[d]] gets the masked logits, for d < ndec (a device scalar). Grid is the fixed
    `cap`, so the launch is CUDA-graph safe: nothing here depends on host state."""
    E = logits.shape[1]
    assert out.shape == logits.shape and out.is_contiguous() and logits.is_contiguous()
    assert bank.dtype == torch.int8 and bank.shape[1] == E and bank.is_contiguous()
    _decode_kernel[(cap,)](logits, out, bank, rows, slots, ndec, count, E, float(rho),
                           BLOCK=_block(E), NSWAPS=int(swaps), num_warps=1)


def chunk_scan(lg, res0, R, rho, swaps, count):
    """Walk one chunk lg [n,E]. res0: int8 [E] state before the chunk, or None for an
    unseeded chunk (cold fill = top-R of lg[0], done in torch so ties match the reference).
    Returns int8 [n,E]: the resident mask AFTER each token (out[0] = cold fill when
    unseeded). `count` is the device counter to add swaps to (pass a scratch tensor for
    prefill observation, which is not weight movement)."""
    n, E = lg.shape
    lg = lg.contiguous()
    if res0 is None:
        res0 = torch.zeros(E, dtype=torch.int8, device=lg.device)
        res0.scatter_(0, lg[0].float().topk(R).indices, 1)
        t0 = 1
    else:
        t0 = 0
    out = torch.empty(n, E, dtype=torch.int8, device=lg.device)
    _chunk_scan_kernel[(1,)](lg, res0, out, count, n, t0, E, float(rho),
                             BLOCK=_block(E), NSWAPS=int(swaps), num_warps=1)
    return out


# ---------------------------------------------------------------------------
# Plain-torch references (the definition the kernels are held to).
# ---------------------------------------------------------------------------
def reference_step(lt, resident, rho, swaps=1):
    """lt [B,E] any float dtype, resident bool [B,E] -> (resident', swaps_done [B])."""
    lt = lt.float()
    E = lt.shape[-1]
    done = torch.zeros(lt.shape[0], dtype=torch.long, device=lt.device)
    for _ in range(swaps):
        nom_val, nom_i = lt.masked_fill(resident, NEG).max(dim=-1)
        worst_val, _ = lt.masked_fill(~resident, POS).min(dim=-1)
        do_swap = (nom_val > worst_val + rho).unsqueeze(-1)
        evict_i = lt.masked_fill(~resident, POS).argmin(dim=-1)
        evicted = torch.nn.functional.one_hot(evict_i, E).bool() & do_swap
        nominee = torch.nn.functional.one_hot(nom_i, E).bool() & do_swap
        resident = (resident & ~evicted) | nominee
        done += do_swap[:, 0].long()
    return resident, done


def reference_scan(lg, res0, R, rho, swaps=1):
    """Token-by-token version of chunk_scan. Returns (bool [n,E], swaps_done)."""
    n, E = lg.shape
    if res0 is None:
        res = torch.zeros(1, E, dtype=torch.bool, device=lg.device)
        res.scatter_(1, lg[0:1].float().topk(R, dim=-1).indices, True)
        t0 = 1
    else:
        res = res0.bool().view(1, E)
        t0 = 0
    out = torch.zeros(n, E, dtype=torch.bool, device=lg.device)
    if t0 == 1:
        out[0] = res[0]
    total = 0
    for t in range(t0, n):
        res, d = reference_step(lg[t:t + 1], res, rho, swaps)
        out[t] = res[0]
        total += int(d)
    return out, total
