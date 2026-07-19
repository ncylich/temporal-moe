#!/usr/bin/env python3
"""Temporal-MoE machinery for the Mac MLX serving benchmark (PLAN.md Phase 2).

Cold pool = each MoE block's existing [E, out, in] quantized expert tensors
(gate_proj / up_proj / down_proj, each weight+scales+biases). In temporal mode
the cold pool is NEVER on the compute path. Hot slots = separate [R, ...] arrays
(R = num_experts_per_tok by default) that the expert GEMM reads via
`mx.gather_qmm` with remapped slot indices.

Semantics mirror the CUDA fork exactly (llamacpp-bench/temporal.cu, k_residency):

  * deploy (force1 path): <=1 swap per layer/token at p=1.0.
      - nominee (which expert swaps in) = LOWEST-INDEX selected expert that is
        not resident (temporal.cu: `if (sh_sel[e] && loc[e]<0) atomicMin(&s_nom,e)`).
      - victim (which slot is evicted) = LOWEST-INDEX slot holding a
        non-selected expert, or an empty slot -- NOT LRU
        (`if (re<0 || !sh_sel[re]) atomicMin(&s_vic,s)`).
      - id-remap: a selected expert that is resident maps to its slot; a
        selected expert that is still NON-resident maps to slot 0
        (`remap_out[off] = loc[e] >= 0 ? loc[e] : 0`).
      - drive rate: the fork benchmarks deploy at TEMPORAL_SWAP_PROB=1.0
        (~1 swap-copy per layer per token of real bytes). Our collapsed q4
        random router produces a natural nominee on only ~half the layers, so
        the deploy path is BRANCHLESS: when no swap is needed it copies the
        victim slot's occupant onto itself (cold->hot, real bytes, unchanged
        semantics). Byte rate = exactly 1 expert/layer/token, and the output
        still equals the natural masked-routing reference bit-for-bit.

  * lazy_full (budget=R fallback, TEMPORAL_UNIFIED_NOFORCE1): swap every
      selected non-resident expert; eviction is a monotonic forward scan over
      slots picking the lowest-index non-selected slot each time. With R=k this
      makes all k selected experts resident -> exact full-MoE. This is the
      fork's "NOFORCE1 bit-identical" proof of the swap/remap/GEMM infra.
      (Gate-only mode; stays host-side/slow.)

  * noswap (setup b analog): machinery on, ZERO swaps, static initial table;
      selected ids remapped through loc (resident -> its slot, non-resident ->
      slot 0). Zero bytes moved -- pure machinery overhead.

  * floor (TEMPORAL_SWAP_N analog, vanilla-offload floor): a TIMING emulation.
      Per layer/token, physically copy EXACTLY N experts cold->hot with source
      ids CYCLED across calls (defeat caching), victim slots by the same
      lowest-index-non-selected eviction rule; then the expert GEMM reads all R
      slots (rhs_indices = arange(R)) so it has a real data dependency on every
      one of the N copies -- lazy eval cannot elide or reorder them. Numerical
      output is not meaningful (it is a floor); the point is faithful bytes/time.

Perf architecture (mirrors the fork's zero-host-sync design): residency tables
(slot_of [R], loc [E]) live ON-DEVICE as mx int32 arrays; nominee/victim/remap
are computed with branchless mx ops inside `mx.compile`d per-layer step
functions, and the cold->hot copies use device-computed indices. There is no
GPU->CPU sync anywhere on the decode path, so bench_decode's async_eval token
pipeline stays intact (the earlier host-side numpy residency forced 45 syncs
per token and halved throughput). lazy_full / deploy_ref remain host-side --
they are correctness gates, not benchmarks.

Ambiguity note: temporal.cu computes the remap for ALL selected ids from the
post-swap `loc[]` table (fused into k_residency). We reproduce that: remap is
built after the residency update, from the updated table.
"""
import os
from functools import lru_cache

import mlx.core as mx
import numpy as np

from mlx_lm.models.activations import swiglu

PROJ = ("gate_proj", "up_proj", "down_proj")


# ---------- compiled on-device residency steps (one cached fn per shape) ----------
@lru_cache(maxsize=None)
def _deploy_step(E, R):
    ridx = mx.arange(R, dtype=mx.int32)

    def step(slot, sel):
        sel = sel.astype(mx.int32)
        m = sel[:, None] == slot[None, :]          # [k,R] membership
        resident = mx.any(m, axis=-1)              # [k] selected expert resident?
        nom = mx.min(mx.where(resident, E, sel))   # lowest-index selected non-res
        slot_sel = mx.any(m, axis=0)               # [R] slot holds a selected?
        vic = mx.min(mx.where(slot_sel, R, ridx))  # lowest-index non-selected slot
        do = (nom < E) & (vic < R)
        vicc = mx.minimum(vic, R - 1)
        # branchless: no natural swap -> re-copy the occupant (real bytes, no-op math)
        src = mx.where(do, nom, mx.take(slot, vicc))
        new_slot = mx.where((ridx == vicc) & do, nom, slot)
        # VIRTUAL SLOTS: effective EXPERT ids. resident (or just-swapped-in
        # nominee) -> itself; still-non-resident -> the expert in slot 0 (the
        # fork's remap-to-slot-0 rule seen through the slot contents).
        now_res = resident | (do & (sel == nom))
        eff = mx.where(now_res, sel, mx.take(new_slot, 0))
        return new_slot, src.reshape(1), eff

    return mx.compile(step)


@lru_cache(maxsize=None)
def _noswap_step(E, R):
    del E, R  # static table; keyed for cache symmetry

    def step(slot, sel):
        sel = sel.astype(mx.int32)
        resident = mx.any(sel[:, None] == slot[None, :], axis=-1)
        return mx.where(resident, sel, mx.take(slot, 0))

    return mx.compile(step)


def expert_glu(x, tensors, remap, group_size, bits, mode):
    """SwiGLU expert GEMM over hot slots (mirrors SwitchGLU op order): gather_qmm
    up_proj then gate_proj, swiglu, then down_proj. `tensors` maps each
    projection name to its (w, s, b) triple. The op order is bitwise-load-bearing
    (the exactness gates check it) -- do not reorder."""
    xe = mx.expand_dims(x, (-2, -3))

    def qmm(name):
        w, s, b = tensors[name]
        return mx.gather_qmm(xe, w, s, b, rhs_indices=remap, transpose=True,
                             group_size=group_size, bits=bits, mode=mode)

    x_up = qmm("up_proj")
    x_gate = qmm("gate_proj")
    w, s, b = tensors["down_proj"]
    out = mx.gather_qmm(swiglu(x_gate, x_up), w, s, b, rhs_indices=remap,
                        transpose=True, group_size=group_size, bits=bits,
                        mode=mode)
    return out.squeeze(-2)


class TemporalLayer:
    """Per-MoE-block temporal state + forward. Attached as `block.temporal`."""

    def __init__(self, block, ctrl, R):
        self.block = block
        self.ctrl = ctrl
        sm = block.switch_mlp
        self.E = block.num_experts
        self.k = block.top_k
        self.R = R
        # quant params (identical across the 3 projections)
        gp = sm.gate_proj
        self.gs, self.bits, self.mode = gp.group_size, gp.bits, gp.mode
        # cold pool references (never mutated)
        self.cold = {
            name: (getattr(sm, name).weight,
                   getattr(sm, name).scales,
                   getattr(sm, name).biases)
            for name in PROJ
        }
        self.expert_bytes = sum(
            t[0:1].nbytes for tensors in self.cold.values() for t in tensors)
        # flat cold pool [E, expert_bytes] uint8 for bench-mode staging: one
        # contiguous take per layer instead of 9 strided ones (the dispatch
        # fix; byte count identical). Built only for bench modes.
        self.cold_flat = None
        if ctrl.mode in ("deploy", "noswap", "floor"):
            E = self.E
            self.cold_flat = mx.concatenate(
                [mx.view(t.reshape(E, -1), dtype=mx.uint8)
                 for name in PROJ for t in self.cold[name]], axis=1)
            mx.eval(self.cold_flat)
            assert self.cold_flat.shape == (E, self.expert_bytes)
        self.hot = None
        self.reset()

    # ---- state ----
    def reset(self):
        R, E = self.R, self.E
        # host tables (lazy_full / deploy_ref gate modes only)
        self.slot_of = np.arange(R, dtype=np.int64)          # slot -> expert id
        self.loc = np.full(E, -1, dtype=np.int64)            # expert -> slot
        self.loc[:R] = np.arange(R)
        # device table (noswap / deploy / floor bench modes): slot -> expert.
        # (loc is derivable from slot membership; device modes are E-array-free.)
        self.slot_mx = mx.arange(R, dtype=mx.int32)
        self._cycle = 0
        # hot slots = fresh copy of cold[:R] (so table & bytes are consistent)
        self.hot = {}
        for name in PROJ:
            w, s, b = self.cold[name]
            self.hot[name] = [mx.array(w[:R]), mx.array(s[:R]), mx.array(b[:R])]
        mx.eval(self.slot_mx,
                *[a for t in self.hot.values() for a in t])

    # ---- host residency policies (gate modes; mirror temporal.cu k_residency) ----
    def _resid_deploy(self, selset):
        """force1: <=1 swap. nominee/victim both lowest-index. Returns swaps."""
        loc, slot, R, E = self.loc, self.slot_of, self.R, self.E
        nominee = -1
        for e in range(E):
            if loc[e] < 0 and e in selset:
                nominee = e
                break
        victim = -1
        for s in range(R):
            re = slot[s]
            if re < 0 or re not in selset:
                victim = s
                break
        if nominee >= 0 and victim >= 0:
            old = slot[victim]
            if old >= 0:
                loc[old] = -1
            slot[victim] = nominee
            loc[nominee] = victim
            return [(nominee, victim)]
        return []

    def _resid_lazy(self, selset):
        """budget=R serial scan, monotonic evict pointer. Returns swaps."""
        loc, slot, R, E = self.loc, self.slot_of, self.R, self.E
        sp = 0
        budget = R
        swaps = []
        for e in range(E):
            if budget <= 0:
                break
            if e not in selset or loc[e] >= 0:
                continue
            victim = -1
            while sp < R:
                re = slot[sp]
                if re < 0 or re not in selset:
                    victim = sp
                    break
                sp += 1
            if victim < 0:
                break
            sp += 1
            old = slot[victim]
            if old >= 0:
                loc[old] = -1
            slot[victim] = e
            loc[e] = victim
            swaps.append((e, victim))
            budget -= 1
        return swaps

    # ---- copies (real cold->hot bytes; counted) ----
    def _copy(self, pairs):
        """Host-indexed copies (gate modes). pairs: list[(expert_id, slot)]."""
        if not pairs:
            return 0
        exp = mx.array([e for e, _ in pairs], dtype=mx.uint32)
        vic = mx.array([s for _, s in pairs])
        return self._copy_dev(exp, vic)

    def _copy_dev(self, srcs, vics):
        """Host/device-indexed cold->hot SLOT copies (lazy_full gate mode only).
        Scatters into the hot arrays; proves the copy+scatter infra exact."""
        for name in PROJ:
            cw, cs, cb = self.cold[name]
            hw, hs, hb = self.hot[name]
            hw[vics] = mx.take(cw, srcs, axis=0)
            hs[vics] = mx.take(cs, srcs, axis=0)
            hb[vics] = mx.take(cb, srcs, axis=0)
        return self.expert_bytes * int(srcs.shape[0])

    def _stage_disk(self, x, n):
        """Regime-2 tier crossing, minimum-overhead form: blocking threaded
        preadv of n experts' bytes from the >RAM disk pool directly into
        preallocated numpy ring buffers. Causality is enforced by control flow
        (the caller has already mx.eval'd this layer's routing, and the reads
        BLOCK before graph building continues), so no staging mx.array or
        mx.depends is needed -- zero graph overhead, zero extra RAM copies.
        The serial fetch-then-compute ordering is the same for floor and
        deploy (conservative for deploy, which a real async engine could
        partially overlap)."""
        ctrl = self.ctrl
        if ctrl.diag == "nocopy":
            return x, 0
        eb, E_d, L = self.expert_bytes, ctrl.disk_E, self.disk_idx
        buf = ctrl.disk_ring[ctrl.disk_ring_i]
        ctrl.disk_ring_i = (ctrl.disk_ring_i + 1) % len(ctrl.disk_ring)
        if n == 1:
            # single-expert fetch (deploy): parallel sub-reads of one expert.
            # Issue + block here (sync deploy); router-early reuses the same
            # issue via _issue_disk_n1 but waits later (overlaps attention).
            futures, _ = self._issue_disk_n1(buf)
            for f in futures:
                f.result()
        else:
            offs = [((L * E_d + (self._dcycle + i * 7919) % E_d) * eb, i * eb)
                    for i in range(n)]
            self._dcycle = (self._dcycle + n * 7919 + 1) % E_d
            def rdn(p_):
                o, at = p_
                got = os.preadv(ctrl.disk_fd, [memoryview(buf)[at:at + eb]], o)
                assert got == eb, f"short preadv at offset {o}: {got} != {eb}"
                return got
            list(ctrl.disk_ex.map(rdn, offs))
        return x, eb * n

    def _issue_disk_n1(self, buf):
        """Non-blocking issue of the single-expert split pread into `buf`.
        Returns (futures, bytes). Same split-read internals as _stage_disk's
        n==1 branch, but ISSUE and WAIT are separated so router-early can start
        the reads, run GPU attention, then wait -- overlapping the two. The
        sync deploy path calls this and blocks immediately (identical bytes /
        offset stream). Advancing the ring buffer stays with the caller."""
        ctrl = self.ctrl
        eb, E_d, L = self.expert_bytes, ctrl.disk_E, self.disk_idx
        off = (L * E_d + (self._dcycle % E_d)) * eb
        self._dcycle = (self._dcycle + 7919) % E_d
        sub = ctrl.disk_split
        step = -(-eb // sub)
        parts = [(off + j * step, j * step, min(step, eb - j * step))
                 for j in range(sub)]
        def rd1(p_):
            o, at, ln = p_
            got = os.preadv(ctrl.disk_fd, [memoryview(buf)[at:at + ln]], o)
            assert got == ln, f"short preadv at offset {o}: {got} != {ln}"
            return got
        return [ctrl.disk_ex.submit(rd1, p) for p in parts], eb * 1

    def _stage(self, x, srcs, n):
        """Bench-mode tier-crossing copy: gather `srcs` experts' bytes cold->
        fresh staging buffers (the same read+write traffic as a slot copy, none
        of MLX's functional-scatter artifact), then order them BEFORE this
        layer's expert GEMM via mx.depends on x. A scatter-free MLX offload
        engine would compute from exactly such gathered buffers. Returns
        (x_with_dependency, bytes_moved)."""
        if self.ctrl.diag == "nocopy":   # fork TEMPORAL_UNIFIED_NOCOPY analog:
            return x, 0                  # timing decomposition only
        st = self.ctrl.copy_stream
        if st is None:
            staged = mx.take(self.cold_flat, srcs, axis=0)
        else:
            with mx.stream(st):
                staged = mx.take(self.cold_flat, srcs, axis=0)
        if self.ctrl.retain_bytes:
            # verification knobs (see tests/v1_slc_probe.py + RESULTS.md appendix)
            self.ctrl.retained.append(staged)
            self.ctrl.retained_b += staged.nbytes
            while self.ctrl.retained_b > self.ctrl.retain_bytes:
                self.ctrl.retained_b -= self.ctrl.retained.popleft().nbytes
        return mx.depends([x], [staged])[0], self.expert_bytes * n

    # ---- expert GEMM over HOT SLOTS (mirrors SwitchGLU op order) ----
    def _expert_glu(self, x, remap):
        return expert_glu(x, self.hot, remap, self.gs, self.bits, self.mode)

    # ---- forward for one decode token (x: [1,1,H]) ----
    def __call__(self, x):
        inds, scores = self.block.route(x)          # true top-k routing (unchanged)
        mode = self.ctrl.mode

        if mode in ("deploy_ref", "lazy_full"):
            # host-side gate modes: sync on the selection (slow; correctness only)
            sel_np = np.array(inds[0, 0]).astype(np.int64)
            selset = set(sel_np.tolist())
            if mode == "deploy_ref":
                # G2b-ii reference emulator: SAME deploy residency rule, but the
                # expert GEMM is computed straight from the COLD pool via
                # effective ids -- no hot slots, no byte copies.
                self._resid_deploy(selset)
                eff = np.where(self.loc[sel_np] >= 0, sel_np, self.slot_of[0])
                y = self.block.switch_mlp(x, mx.array(eff.reshape(1, 1, self.k)))
                return (y * scores[..., None]).sum(axis=-2)
            swaps = self._resid_lazy(selset)
            self.ctrl.copied_bytes += self._copy(swaps)
            r = np.where(self.loc[sel_np] >= 0, self.loc[sel_np], 0)
            remap = mx.array(r.reshape(1, 1, self.k))
            y = self._expert_glu(x, remap)
            return (y * scores[..., None]).sum(axis=-2)

        # device modes: zero host syncs, virtual slots (effective EXPERT ids
        # over the cold pool -- op-order-identical to the reference emulator),
        # tier-crossing bytes as staged copies ordered before the GEMM.
        sel = inds.reshape(self.k)
        if mode == "deploy":
            if self.ctrl.diag == "residcpu":   # experiment: tiny residency ops on CPU stream
                with mx.stream(mx.cpu):
                    self.slot_mx, src, eff = _deploy_step(
                        self.E, self.R)(self.slot_mx, sel)
            else:
                self.slot_mx, src, eff = _deploy_step(
                    self.E, self.R)(self.slot_mx, sel)
            if self.ctrl.disk_fd is not None:
                mx.eval(src)                  # causality: swap decision first
                x, nb = self._stage_disk(x, 1)
            else:
                x, nb = self._stage(x, src.astype(mx.uint32), 1)
            self.ctrl.copied_bytes += nb
        elif mode == "noswap":
            eff = _noswap_step(self.E, self.R)(self.slot_mx, sel)
        elif mode == "floor" and self.ctrl.disk_fd is not None:
            N = self.ctrl.N
            mx.eval(sel)                      # causality: routing first (also
            if N > 0:                         #   the n0 sync-loop baseline)
                x, nb = self._stage_disk(x, N)
                self.ctrl.copied_bytes += nb
            eff = mx.broadcast_to(mx.arange(self.R, dtype=mx.int32), (1, 1, self.R))
        elif mode == "floor":
            N, E = self.ctrl.N, self.E
            if N > 0:
                st0 = 0 if self.ctrl.same_window else self._cycle % max(1, E - N)
                self._cycle += N
                # cycled sources, made DATA-DEPENDENT on this layer's router
                # output (+ sel[0]*0): fetch-on-miss causality. Without this the
                # indices are graph constants and Metal hoists the copies into
                # earlier layers' compute = illegal prefetch for a floor.
                srcs = (mx.arange(st0, st0 + N, dtype=mx.int32)
                        + mx.take(sel, 0).astype(mx.int32) * 0).astype(mx.uint32)
                x, nb = self._stage(x, srcs, N)
                self.ctrl.copied_bytes += nb
            # timing emulation: GEMM shape/cost identical for any R ids
            eff = mx.broadcast_to(mx.arange(self.R, dtype=mx.int32), (1, 1, self.R))
        else:
            raise ValueError(f"unknown temporal mode {mode!r}")
        y = self.block.switch_mlp(x, eff.reshape(1, 1, -1).astype(mx.uint32))
        return (y * scores[..., None]).sum(axis=-2)

    # ---- ROUTER-EARLY split forward (disk deploy only; driven by the decoder
    # block in model.py so it can interleave attention between issue and wait).
    # The router + residency decision + expert-fetch ISSUE run on the
    # PRE-attention input; attention runs while the pread threads are in flight;
    # the expert GEMM runs POST-attention on the post-attention hidden state but
    # with this pre-attention routing decision. Because the disk bytes never
    # feed the math (the GEMM reads the cold pool through effective ids, exactly
    # as the sync deploy device path does), overlap changes only TIMING -- the
    # logits are identical to the TEMPORAL_EARLY_NOOVERLAP control bit-for-bit.
    def route_issue(self, r_in):
        """Steps 1-3: route on the pre-attention (post_attention_layernorm'd)
        residual, run the branchless deploy residency step, issue the overlapped
        single-expert fetch. Returns (eff, scores, futures, bytes) for finish().
        TEMPORAL_EARLY_NOOVERLAP=1 blocks the fetch here (sequential control)."""
        ctrl = self.ctrl
        inds, scores = self.block.route(r_in)
        sel = inds.reshape(self.k)
        # same branchless self-copy drive as the sync deploy path
        self.slot_mx, src, eff = _deploy_step(self.E, self.R)(self.slot_mx, sel)
        mx.eval(src)                      # swap decision first (attn not yet built)
        if ctrl.diag == "nocopy":
            return eff, scores, [], 0
        buf = ctrl.disk_ring[ctrl.disk_ring_i]
        ctrl.disk_ring_i = (ctrl.disk_ring_i + 1) % len(ctrl.disk_ring)
        futures, nb = self._issue_disk_n1(buf)
        if ctrl.early_nooverlap:          # overlap-isolation control: block now
            for f in futures:
                f.result()
            futures = []
        return eff, scores, futures, nb

    def expert_finish(self, h_norm, eff, scores, futures, nb):
        """Steps 5-6: wait the (already-overlapped) fetch, count bytes, run the
        expert GLU on the POST-attention hidden state with the pre-attention
        routing decision."""
        for f in futures:
            f.result()
        self.ctrl.copied_bytes += nb
        y = self.block.switch_mlp(h_norm, eff.reshape(1, 1, -1).astype(mx.uint32))
        return (y * scores[..., None]).sum(axis=-2)


class TemporalController:
    """Enables temporal mode on all MoE blocks of a loaded Model.

    mode: 'noswap' | 'deploy' | 'lazy_full' | 'floor' | 'deploy_ref'.
    R defaults to num_experts_per_tok (k); pass R=E for the G2a identity test.
    N: floor swap count (ignored unless mode=='floor').
    copy_stream: optional mx stream for the cold->hot copies (deploy overlap,
    Phase 4b); None = default stream (synchronous ordering).
    """

    def __init__(self, model, mode, R=None, N=0, copy_stream=None):
        import os
        from collections import deque
        self.diag = os.environ.get("TEMPORAL_DIAG", "")
        # cache-verification knobs:
        # TEMPORAL_STAGE_RETAIN_MB: keep staged outputs alive until this many MB
        #   -> allocator must rotate write targets over >SLC memory.
        # TEMPORAL_FLOOR_SAMEWINDOW=1: freeze floor sources at window 0
        #   (short read-reuse distance -> read-cache test).
        # TEMPORAL_ALIAS_POOLS=1: all layers share layer-0's flat pool
        #   (with SAMEWINDOW: ~10 MB read set, the SLC-detectability control).
        self.retain_bytes = None
        if os.environ.get("TEMPORAL_STAGE_RETAIN_MB"):
            self.retain_bytes = int(os.environ["TEMPORAL_STAGE_RETAIN_MB"]) * 2**20
        self.retained = deque()
        self.retained_b = 0
        self.same_window = os.environ.get("TEMPORAL_FLOOR_SAMEWINDOW", "") == "1"
        self.alias_pools = os.environ.get("TEMPORAL_ALIAS_POOLS", "") == "1"
        # router-early architectural variant (bench setup deploy_early): the
        # decoder block routes + decides residency + ISSUES the expert fetch on
        # the pre-attention input, runs attention while the fetch is in flight,
        # then runs the experts post-attention. bench_decode sets router_early
        # True. TEMPORAL_EARLY_NOOVERLAP=1 blocks the fetch at issue time
        # (sequential) -- the overlap-isolation control (same math, no overlap).
        self.router_early = False
        self.early_nooverlap = os.environ.get("TEMPORAL_EARLY_NOOVERLAP", "") == "1"
        # regime-2 disk tier (bigger-than-RAM cold pool): TEMPORAL_DISK_POOL=
        # <file>, TEMPORAL_DISK_QD=<threads>. Tier-crossing fetches become real
        # SSD preads (F_NOCACHE fd; the pool exceeds RAM so the page cache
        # cannot hold it) issued only AFTER the layer's routing is materialized
        # (per-layer mx.eval = fetch-on-miss causality; its cost is measured by
        # the floor_n=0 baseline under the same sync loop). Sources cycled with
        # a large prime stride = worst-case no-temporal-locality miss stream.
        self.disk_path = os.environ.get("TEMPORAL_DISK_POOL", "")
        self.disk_qd = int(os.environ.get("TEMPORAL_DISK_QD", "8"))
        self.disk_fd = None
        self.disk_E = 0
        self.disk_ex = None
        self.mode = mode
        self.N = N
        self.copied_bytes = 0
        self.copy_stream = copy_stream
        self._model = model
        k = model.args.num_experts_per_tok
        self.R = k if R is None else R
        self.layers = []
        for layer in model.model.layers:
            tl = TemporalLayer(layer.mlp, self, self.R)
            layer.mlp.temporal = tl
            self.layers.append(tl)
        if self.alias_pools and self.layers and self.layers[0].cold_flat is not None:
            for tl in self.layers[1:]:
                tl.cold_flat = self.layers[0].cold_flat
        # default write-target span = ONE TOKEN's staged bytes, i.e. what a real
        # engine's persistent per-layer hot buffers occupy (45 distinct write
        # targets). Prevents MLX's recycled staging buffer from staying
        # SLC-resident and under-charging the floor's DRAM writes (~7% at n16;
        # verified in the RESULTS.md cache-verification appendix).
        if self.retain_bytes is None and self.layers:
            per_copy = self.layers[0].expert_bytes * (N if mode == "floor" else 1)
            self.retain_bytes = per_copy * len(self.layers)
        if self.disk_path and mode in ("deploy", "floor"):
            import fcntl
            from concurrent.futures import ThreadPoolExecutor
            self.disk_fd = os.open(self.disk_path, os.O_RDONLY)
            fcntl.fcntl(self.disk_fd, 48, 1)   # F_NOCACHE
            fsz = os.fstat(self.disk_fd).st_size
            eb = self.layers[0].expert_bytes
            stride = len(self.layers) * eb
            assert fsz % stride == 0, (
                f"disk pool {self.disk_path!r} size {fsz} not a multiple of "
                f"n_layers*expert_bytes ({len(self.layers)}*{eb}={stride})")
            self.disk_E = fsz // stride
            self.disk_ex = ThreadPoolExecutor(max_workers=self.disk_qd)
            self.disk_split = int(os.environ.get("TEMPORAL_DISK_SPLIT", "8"))
            ring_sz = max(1, N) * eb
            self.disk_ring = [np.empty(ring_sz, dtype=np.uint8) for _ in range(2)]
            self.disk_ring_i = 0
            for i, tl in enumerate(self.layers):
                tl.disk_idx = i
                tl._dcycle = 0

    @property
    def n_moe_layers(self):
        return len(self.layers)

    def reset(self):
        self.copied_bytes = 0
        self.retained.clear()
        self.retained_b = 0
        for tl in self.layers:
            tl.reset()

    def disable(self):
        for layer in self._model.model.layers:
            layer.mlp.temporal = None
        if self.disk_ex is not None:
            self.disk_ex.shutdown(wait=False)
            self.disk_ex = None
        if self.disk_fd is not None:
            try:
                os.close(self.disk_fd)
            except OSError:
                pass
            self.disk_fd = None
