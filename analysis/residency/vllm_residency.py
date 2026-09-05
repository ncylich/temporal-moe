#!/usr/bin/env python3
"""Decode-time residency under continuous batching (vLLM glue, engine-agnostic core).

vLLM flattens every scheduled request into one token stream per model step: prefill
chunks (n>1 or a 1-token tail) and single-token decode steps, interleaved, membership
changing step to step. The core here is deliberately tiny: a per-step span list
(req_id, n_tokens, is_prefill) published by a ~10-line runner patch, and `apply()`,
which walks spans over the flattened router logits keying decode_state by
(req_id, layer): prefill spans are observed free (protocol), decode spans get one
reference _step and a mask.

Correctness under scheduling churn is closed-form: state is keyed by request, never by
batch row, so compaction/reordering cannot mix streams; preemption in vLLM v1 discards
KV and replays the prefill, which rebuilds state from scratch, so pruning state for
requests absent from the current step is always safe (done in the runner patch).
test_vllm_walker.py simulates an adversarial schedule and asserts mask equality with
the per-request reference scan.
"""
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_state as DS                                            # noqa: E402
from decode_state import DEC                                         # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from temporal import temporal_router as TR                           # noqa: E402
import residency_kernels as RK                                       # noqa: E402

# Walker selection. "fast" (default as of 2026-08-28): fused Triton kernels over
# persistent device buffers, CUDA-graph safe (see residency_kernels.py). "slots"
# and "dict" are the earlier python walkers, kept for comparison.
_WALKER = os.environ.get("TEMPORAL_WALKER", "fast")

STEP = {"spans": None}     # [(req_id, n_tokens, is_prefill)] for the current model step


def set_step(spans):
    STEP["spans"] = spans
    STEP["plan"] = None                         # rebuilt lazily by the slotted path
    if spans is not None:                       # prune state of departed requests (safe:
        live = {sp[0] for sp in spans}          # preemption replays the prefill)
        # tuple keys only: state is keyed (req_id, layer); the slotted path parks a
        # non-tuple epoch marker here and `k[0]` on a str is its first CHARACTER,
        # which silently matched nothing and pruned the marker every step
        for key in [k for k in DEC["state"]
                    if isinstance(k, tuple) and k[0] not in live]:
            del DEC["state"][key]
        if SL["rows"]:                          # free the slots those requests held
            for req in [r for r in SL["rows"] if r not in live]:
                SL["free"].append(SL["rows"].pop(req))
                SL["seeded"].discard(req)
        if FAST["reqrow"]:
            for req in [r for r in FAST["reqrow"] if r not in live]:
                FAST["free"].append(FAST["reqrow"].pop(req))
                FAST["seeded"].discard(req)
    if _WALKER == "fast":
        _fast_set_step(spans)
    if _WALKER == "cache_bias" and spans is not None:
        import cache_bias as CB
        CB.prune({sp[0] for sp in spans})


# ---------------------------------------------------------------------------
# Slotted state (TEMPORAL_WALKER=slots).
#
# The per-request dict above costs more than the residency rule it serves: every
# layer of every decode step rebuilt the batch with `torch.cat` over D one-row
# tensors, then sliced the result back into D fresh views and reassigned the
# dict. Measured at D=64, 36 layers: 0.41 ms/layer, of which 60% is that
# bookkeeping -- 14.6 ms per decode step against 2.0 ms for the same rule over a
# persistent buffer (7.3x).
#
# Here each request holds a row in one [cap,E] resident tensor PER LAYER, and the
# row index is layer-independent, so the O(D) python work happens ONCE per step
# (in the plan) rather than once per layer. Per layer the hot path is then
# index_select -> step -> index_copy_, all tensor ops.
#
# Same correctness argument as the dict: state is keyed by request, never by
# batch row, so compaction/reordering cannot mix streams. A slot is freed only
# when its request is absent from the step (which, per the module docstring, is
# exactly when replaying its prefill would rebuild the state anyway), and a
# freshly assigned row is always written by a prefill observe or a cold fill
# before it is read. `refresh` is not carried: min_logit eviction never reads it
# (see temporal_router._minlogit_step).
# ---------------------------------------------------------------------------
SL = {"rows": {}, "free": [], "next": 0, "res": {}, "cap": 0, "epoch": None,
      "seeded": set()}


def _slots_live():
    """Detect the harnesses' `DEC["state"].clear()` between arms and reset with it:
    a stale resident set leaking across arms would silently corrupt a whole cell."""
    if DEC["state"].get("__epoch__") != SL["epoch"] or SL["epoch"] is None:
        SL.update(rows={}, free=[], next=0, res={}, cap=0, epoch=object(),
                  seeded=set())
        DEC["state"]["__epoch__"] = SL["epoch"]


def _row(req):
    r = SL["rows"].get(req)
    if r is None:
        r = SL["free"].pop() if SL["free"] else SL["next"]
        if r == SL["next"]:
            SL["next"] += 1
        SL["rows"][req] = r
    return r


def _bank(layer, E, device):
    """The [cap,E] resident tensor for one layer, grown (never shrunk) to fit."""
    need = max(SL["next"], 1)
    t = SL["res"].get(layer)
    if t is None or t.shape[0] < need:
        cap = max(64, need * 2)
        nt = torch.zeros(cap, E, dtype=torch.bool, device=device)
        if t is not None:
            nt[: t.shape[0]].copy_(t)
        SL["res"][layer] = nt
        SL["cap"] = cap
        t = nt
    return t


def _plan(device):
    """Per-STEP work list, built once and reused by every layer: the O(D) python
    (span walk, slot assignment, index tensors) must not run 36 times per token."""
    p = STEP.get("plan")
    if p is not None:
        return p
    # `seeded` is layer-uniform: every layer sees the same spans in a step, so a
    # request that needs a cold fill needs it at every layer. The decision is
    # taken once here and frozen for the step (the plan is cached), then the
    # request is marked seeded so later steps take the hot path.
    dec_rows, dec_slots, pre, cold, replay = [], [], [], [], []
    fresh = []
    o = 0
    for sp in STEP["spans"]:
        req, n, is_prefill = sp[0], sp[1], sp[2]
        seeded = req in SL["seeded"]
        if is_prefill and len(sp) > 3 and sp[3] == 0:     # restart: see the fast walker
            seeded = False
        if is_prefill:
            pre.append((req, o, n, seeded))
            fresh.append(req)
        elif n > 1:
            replay.append((req, o, n, seeded))
            fresh.append(req)
        elif not seeded:
            cold.append((req, o))               # no prefill seen: cold-fill the row
            fresh.append(req)
        else:
            dec_rows.append(o)
            dec_slots.append(_row(req))
        o += n
    for req in fresh:
        _row(req)
        SL["seeded"].add(req)
    p = {"pre": pre, "cold": cold, "replay": replay,
         "dec_rows": torch.tensor(dec_rows, dtype=torch.long, device=device)
         if dec_rows else None,
         "dec_slots": torch.tensor(dec_slots, dtype=torch.long, device=device)
         if dec_rows else None}
    STEP["plan"] = p
    return p


def _apply_slots(layer, router_logits):
    """Slotted decode: index_select -> step -> index_copy_, no per-request python."""
    N, E = router_logits.shape
    _slots_live()
    pl = _plan(router_logits.device)
    bank = _bank(layer, E, router_logits.device)
    out = router_logits.clone()
    with torch.no_grad():
        for req, o, n, seeded in pl["pre"]:     # prefill: observed free, seeds the row
            bank = _bank(layer, E, router_logits.device)
            _observe_into(bank, _row(req), router_logits[o:o + n].float(), seeded)
        for req, o in pl["cold"]:               # cold fill (no prefill seen)
            row = _row(req)
            bank = _bank(layer, E, router_logits.device)
            lt = router_logits[o:o + 1].float()
            res = torch.zeros(1, E, dtype=torch.bool, device=lt.device)
            res.scatter_(1, lt.topk(DEC["R"], dim=-1).indices, True)
            bank[row: row + 1].copy_(res)
            out[o] = out[o].masked_fill(~res[0], float("-inf"))
        for req, o, n, seeded in pl["replay"]:          # preemption replay: step the chunk
            row = _row(req)
            bank = _bank(layer, E, router_logits.device)
            start = 0
            if not seeded:
                lt = router_logits[o:o + 1].float()
                res = torch.zeros(1, E, dtype=torch.bool, device=lt.device)
                res.scatter_(1, lt.topk(DEC["R"], dim=-1).indices, True)
                bank[row: row + 1].copy_(res)
                out[o] = out[o].masked_fill(~res[0], float("-inf"))
                start = 1
            res = bank[row: row + 1]
            for j in range(start, n):
                lt = router_logits[o + j: o + j + 1].float()
                for _ in range(DEC["swaps"]):
                    res = TR._minlogit_step(lt, res)
                out[o + j] = out[o + j].masked_fill(~res[0], float("-inf"))
            bank[row: row + 1].copy_(res)
        if pl["dec_rows"] is not None:          # the hot path: all tensor ops
            ix = pl["dec_slots"]
            rows = pl["dec_rows"]
            lt = router_logits.index_select(0, rows).float()
            res = bank.index_select(0, ix)
            for _ in range(DEC["swaps"]):
                res = TR.step_accel_mask(lt, res)
            bank.index_copy_(0, ix, res)
            out[rows] = out.index_select(0, rows).masked_fill(~res, float("-inf"))
    return out


def _observe_into(bank, row, lg, seeded):
    """Prefill chunk observed free; carries state across chunks (chunked prefill)."""
    t0 = 0
    if not seeded:
        res = torch.zeros(1, lg.shape[-1], dtype=torch.bool, device=lg.device)
        res.scatter_(1, lg[0:1].topk(DEC["R"], dim=-1).indices, True)
        t0 = 1
    else:
        res = bank[row: row + 1]
    for t in range(t0, lg.shape[0]):
        for _ in range(DEC["swaps"]):
            res = TR._minlogit_step(lg[t: t + 1], res)
    bank[row: row + 1].copy_(res)


# ---------------------------------------------------------------------------
# Fast walker (TEMPORAL_WALKER=fast, default).
#
# State: one persistent [cap,E] int8 bank per layer; request -> slot map kept on the
# host. Per STEP (in set_step, outside any CUDA graph): classify spans once, assign
# slots, and copy three small index buffers to the device (rows of the flattened
# token stream that are decode tokens, their slots, and the count). Per LAYER: one
# clone of the router logits and ONE fused kernel launch that steps every decode row
# and writes the masked logits. Prefill / preemption-replay / cold-fill chunks are
# walked by one kernel launch each, in the python part that only runs on eager steps.
#
# CUDA-graph safety: the decode launch depends on no host state (fixed grid = cap,
# live rows gated by the device scalar ndec), so vLLM can capture it into its
# decode-only graphs and replay it while set_step keeps the buffers current. The
# python part is skipped while a stream is capturing; a capture that finds no
# buffers raises rather than allocating from the graph pool. Free arms run the same
# launch with ndec = 0 (exact no-op on the logits).
# ---------------------------------------------------------------------------
FAST = {"cap": int(os.environ.get("TEMPORAL_MAX_SEQS", "1024")),
        "rho": float(os.environ.get("TEMPORAL_RHO", "0")),
        "banks": {}, "rows": None, "slots": None, "ndec": None, "count": None,
        "scratch": None, "stage": None, "dev_stage": None, "epoch": None,
        "reqrow": {}, "free": [], "next": 0, "seeded": set(), "plan": None,
        "swaps_c": None, "verified": {"step": False, "scan": False}}


def _fast_alloc(device):
    if FAST["rows"] is not None:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("[residency] fast-walker buffers must be allocated before "
                           "CUDA-graph capture (no eager step ran first)")
    cap = FAST["cap"]
    FAST["dev_stage"] = torch.zeros(2 * cap + 1, dtype=torch.int32, device=device)
    FAST["rows"] = FAST["dev_stage"][:cap]
    FAST["slots"] = FAST["dev_stage"][cap:2 * cap]
    FAST["ndec"] = FAST["dev_stage"][2 * cap:]
    FAST["stage"] = [torch.zeros(2 * cap + 1, dtype=torch.int32).pin_memory()
                     for _ in range(2)]
    FAST["count"] = torch.zeros(2, dtype=torch.int64, device=device)
    FAST["scratch"] = torch.zeros(2, dtype=torch.int64, device=device)


def _fast_bank(layer, E, device):
    t = FAST["banks"].get(layer)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[residency] state bank for a new layer requested during "
                               "CUDA-graph capture")
        t = torch.zeros(FAST["cap"], E, dtype=torch.int8, device=device)
        FAST["banks"][layer] = t
    return t


def _fast_epoch():
    """The harnesses clear DEC["state"] between arms; zero the persistent state with it."""
    if DEC["state"].get("__fast_epoch__") != FAST["epoch"] or FAST["epoch"] is None:
        FAST.update(reqrow={}, free=[], next=0, seeded=set(), epoch=object(), plan=None)
        DEC["state"]["__fast_epoch__"] = FAST["epoch"]
        for t in FAST["banks"].values():
            t.zero_()
        if FAST["count"] is not None:
            FAST["count"].zero_()


def _fast_row(req):
    r = FAST["reqrow"].get(req)
    if r is None:
        r = FAST["free"].pop() if FAST["free"] else FAST["next"]
        if r == FAST["next"]:
            FAST["next"] += 1
        if r >= FAST["cap"]:
            raise RuntimeError(f"[residency] more live requests than TEMPORAL_MAX_SEQS={FAST['cap']}")
        FAST["reqrow"][req] = r
    return r


def _fast_set_step(spans):
    _fast_epoch()
    if FAST["rows"] is None:
        _fast_alloc(torch.device("cuda", torch.cuda.current_device()))
    cap = FAST["cap"]
    st = FAST["stage"][0]; FAST["stage"].reverse()          # alternate host buffers
    if spans is None or not DEC["on"]:
        FAST["plan"] = None
        st[2 * cap] = 0
        FAST["dev_stage"][2 * cap:].copy_(st[2 * cap:], non_blocking=True)
        return
    if FAST["swaps_c"] is not None and DEC["swaps"] != FAST["swaps_c"]:
        raise RuntimeError("[residency] DEC['swaps'] changed after the decode kernel was "
                           "captured; restart the process for a different swap budget")
    pre, replay, cold, dec_rows, dec_slots = [], [], [], [], []
    o = 0
    for sp in spans:
        req, n, is_prefill = sp[0], sp[1], sp[2]
        seeded = req in FAST["seeded"]
        # A prefill span starting at absolute token 0 is a (re)start: a request preempted at
        # step s can replay at s+1 with no step in which it was absent, so it was never
        # pruned and its slot still holds decode state. Cold-fill regardless of `seeded`.
        if is_prefill and len(sp) > 3 and sp[3] == 0:
            seeded = False
        row = _fast_row(req)
        if is_prefill:
            pre.append((row, o, n, seeded))
        elif n > 1:
            replay.append((row, o, n, seeded))
        elif not seeded:
            cold.append((row, o))
        else:
            dec_rows.append(o); dec_slots.append(row)
        FAST["seeded"].add(req)
        o += n
    D = len(dec_rows)
    if D > cap:
        raise RuntimeError(f"[residency] {D} decode rows > TEMPORAL_MAX_SEQS={cap}")
    if D:
        st[:D] = torch.tensor(dec_rows, dtype=torch.int32)
        st[cap:cap + D] = torch.tensor(dec_slots, dtype=torch.int32)
    st[2 * cap] = D
    FAST["dev_stage"].copy_(st, non_blocking=True)
    FAST["plan"] = {"pre": pre, "replay": replay, "cold": cold, "D": D}


def _apply_fast(layer, router_logits):
    logits = router_logits.contiguous()
    N, E = logits.shape
    if FAST["rows"] is None:                    # profile / warm-up forward before any step
        _fast_alloc(logits.device)
    bank = _fast_bank(layer, E, logits.device)
    out = logits.clone()
    capturing = torch.cuda.is_current_stream_capturing()
    if FAST["swaps_c"] is None:
        FAST["swaps_c"] = int(DEC["swaps"])
    pl = FAST["plan"]
    if not capturing and pl is not None and DEC["on"]:
        NEG = float("-inf")
        with torch.no_grad():
            for row, o, n, seeded in pl["pre"]:        # observed free; only the state moves
                res0 = bank[row] if seeded else None
                m = RK.chunk_scan(logits[o:o + n], res0, DEC["R"], FAST["rho"],
                                  FAST["swaps_c"], FAST["scratch"])
                if not FAST["verified"]["scan"]:
                    ref, _ = RK.reference_scan(logits[o:o + n], res0, DEC["R"], FAST["rho"],
                                               FAST["swaps_c"])
                    if not torch.equal(m.bool(), ref):
                        raise RuntimeError("[residency] chunk_scan DISAGREES WITH REFERENCE -- aborting")
                    FAST["verified"]["scan"] = True
                    print("[residency] chunk_scan verified == reference", flush=True)
                bank[row].copy_(m[-1])
            for row, o, n, seeded in pl["replay"]:     # generated tokens recomputed as a chunk
                res0 = bank[row] if seeded else None
                m = RK.chunk_scan(logits[o:o + n], res0, DEC["R"], FAST["rho"],
                                  FAST["swaps_c"], FAST["count"])
                bank[row].copy_(m[-1])
                out[o:o + n].masked_fill_(m == 0, NEG)
            for row, o in pl["cold"]:                  # no prefill seen: top-R cold fill
                m = RK.chunk_scan(logits[o:o + 1], None, DEC["R"], FAST["rho"],
                                  FAST["swaps_c"], FAST["scratch"])
                bank[row].copy_(m[0])
                out[o].masked_fill_(m[0] == 0, NEG)
    if not capturing and pl is not None and DEC["on"] and pl["D"] and not FAST["verified"]["step"]:
        # one-time bit-exactness gate on the fused decode step (hard, house rule)
        with torch.no_grad():
            D = pl["D"]
            rows = FAST["rows"][:D].long(); slots = FAST["slots"][:D].long()
            ref, _ = RK.reference_step(logits.index_select(0, rows), bank.index_select(0, slots).bool(),
                                       FAST["rho"], FAST["swaps_c"])
            probe_bank = bank.clone(); probe_out = out.clone()
            RK.decode_step(probe_out, logits, probe_bank, FAST["rows"], FAST["slots"], FAST["ndec"],
                           FAST["scratch"], FAST["rho"], FAST["swaps_c"], FAST["cap"])
            got = probe_bank.index_select(0, slots).bool()
            want_out = logits.index_select(0, rows).masked_fill(~ref, float("-inf"))
            if not torch.equal(got, ref) or not torch.equal(probe_out.index_select(0, rows), want_out):
                raise RuntimeError("[residency] decode_step DISAGREES WITH REFERENCE -- aborting")
            FAST["verified"]["step"] = True
            print("[residency] decode_step verified == reference", flush=True)
    RK.decode_step(out, logits, bank, FAST["rows"], FAST["slots"], FAST["ndec"], FAST["count"],
                   FAST["rho"], FAST["swaps_c"], FAST["cap"])
    return out


def swap_stats():
    """(swaps performed, decode rows stepped, swaps per decode token) since the last
    arm reset; on-device counters, one sync here. Prefill observation is not counted."""
    if _WALKER == "cache_bias":
        import cache_bias as CB
        return CB.stats()
    if FAST["count"] is None:
        return 0, 0, 0.0
    sw, tk = (int(x) for x in FAST["count"].tolist())
    return sw, tk, (sw / tk if tk else 0.0)


def apply(layer, router_logits):
    """router_logits [N,E] flattened across spans -> same tensor with non-resident
    experts masked to -inf on DECODE rows only. Prefill rows pass through free.

    All decode rows advance in ONE batched reference _step (the scan's step is batched
    over B); with hundreds of concurrent requests a per-request loop would be
    kernel-launch-bound -- the exact pathology this stack exists to escape."""
    if _WALKER == "fast" and router_logits.is_cuda:
        return _apply_fast(layer, router_logits)
    if _WALKER == "cache_bias":                 # Skliar's cache-conditional experts at their own setting
        import cache_bias as CB
        return CB.apply(layer, router_logits, STEP["spans"], DEC["on"])
    if not DEC["on"] or STEP["spans"] is None:
        return router_logits
    # DEFAULT SLOTS as of 2026-08-24: validated bit-exact + real wall-clock win
    # on three architecturally distinct models (see temporal_router.step_accel
    # and TODO.md section 6). TEMPORAL_WALKER=dict opts back into the path
    # every pre-2026-08-24 grid row was produced on, if a comparison needs it.
    if _WALKER != "dict":
        return _apply_slots(layer, router_logits)
    N = router_logits.shape[0]
    total = sum(sp[1] for sp in STEP["spans"])
    assert total == N, f"span/token mismatch: spans cover {total}, logits have {N}"
    out = router_logits.clone()
    dec_rows, dec_keys, cold, replay = [], [], [], []
    o = 0
    for sp in STEP["spans"]:
        req_id, n, is_prefill = sp[0], sp[1], sp[2]
        start = sp[3] if len(sp) > 3 else 0     # absolute offset of this span
        key = (req_id, layer)
        if is_prefill:
            # partial residency prefill: the original prompt is observed FREE, but
            # previously-generated tokens carry the rule (they were generated under
            # it, so their KV must be built the same way and the walk must advance
            # through them to rebuild the resident set)
            ef = DEC["enforce_from"].get(req_id)
            if ef is not None and start + n > ef:
                nfree = max(0, ef - start)
                if nfree:
                    DS.observe_chunk(key, router_logits[o:o + nfree].unsqueeze(1))
                res = DEC["state"].get(key)
                if res is None:
                    lt0 = router_logits[o + nfree:o + nfree + 1].float()
                    r0 = torch.zeros_like(lt0, dtype=torch.bool)
                    r0.scatter_(1, lt0.topk(DEC["R"], dim=-1).indices, True)
                    f0 = torch.zeros_like(lt0)
                    DEC["state"][key] = (r0, f0)
                resident, refresh = DEC["state"][key]
                with torch.no_grad():
                    for j in range(nfree, n):
                        lt = router_logits[o + j:o + j + 1].float()
                        for _ in range(DEC["swaps"]):
                            resident, refresh = DS.step_accel(lt, resident, refresh,
                                                              use_lru=False)
                        out[o + j] = out[o + j].masked_fill(~resident[0], float("-inf"))
                DEC["state"][key] = (resident, refresh)
            else:
                DS.observe_chunk(key, router_logits[o:o + n].unsqueeze(1))
        elif n > 1:
            # preemption replay: vLLM discarded KV and recomputes the generated
            # tokens as one chunk. Logits are deterministic, so stepping the scan
            # through the chunk rebuilds the original per-token schedule.
            replay.append((key, o, n))
        else:
            if key not in DEC["state"]:
                cold.append((key, o))                # rare: no prefill seen (cold fill)
            else:
                dec_rows.append(o)
                dec_keys.append(key)
        o += n
    for key, row in cold:
        resident = DS.step(key, router_logits[row:row + 1])
        out[row] = out[row].masked_fill(~resident[0], float("-inf"))
    for key, row, n in replay:
        start = 0
        if key not in DEC["state"]:                  # state pruned while preempted:
            r0 = DS.step(key, router_logits[row:row + 1])  # re-seed from first token
            out[row] = out[row].masked_fill(~r0[0], float("-inf"))
            start = 1
        resident, refresh = DEC["state"][key]
        with torch.no_grad():
            for j in range(start, n):
                lt = router_logits[row + j:row + j + 1].float()
                for _ in range(DEC["swaps"]):
                    resident, refresh = DS._step(lt, resident, refresh,
                                                 torch.zeros((), device=lt.device),
                                                 use_lru=False)
                out[row + j] = out[row + j].masked_fill(~resident[0], float("-inf"))
        DEC["state"][key] = (resident, refresh)
    if dec_rows:
        with torch.no_grad():
            lt = router_logits[dec_rows].float()                     # [D, E]
            resident = torch.cat([DEC["state"][k][0] for k in dec_keys])
            refresh = torch.cat([DEC["state"][k][1] for k in dec_keys])
            for _ in range(DEC["swaps"]):
                resident, refresh = DS.step_accel(lt, resident, refresh,
                                                  use_lru=False)
        for j, k in enumerate(dec_keys):
            DEC["state"][k] = (resident[j:j + 1], refresh[j:j + 1])
        out[dec_rows] = out[dec_rows].masked_fill(~resident, float("-inf"))
    return out
