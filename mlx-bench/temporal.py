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
def _deploy_step_fast(E, R, K):
    """The SAME residency step as _deploy_step, hand-fused into ONE custom
    Metal kernel (a literal transcription of the fork's k_residency loops:
    lowest-index nominee, lowest-index non-selected victim, branchless
    self-copy source, remap-to-slot-0). The compiled-ops version costs
    ~95 us/layer of small-kernel dispatch; this is one ~15 us launch.
    Verified equal to _deploy_step decision-for-decision (and the exactness
    gates check the resulting logits bitwise). Used by the bench deploy paths;
    xl.py and the residcpu diagnostic keep the compiled-ops step."""
    src_code = f"""
    if (thread_position_in_grid.x != 0) return;
    const int E_ = {E}; const int R_ = {R}; const int K_ = {K};
    int loc[K_];
    int nom = E_;
    for (int i = 0; i < K_; ++i) {{
        int s = int(sel[i]); int l = -1;
        for (int j = 0; j < R_; ++j) if (slot[j] == s) {{ l = j; break; }}
        if (l < 0 && s < nom) nom = s;
        loc[i] = l;
    }}
    int vic = R_;
    for (int j = 0; j < R_; ++j) {{
        bool selj = false;
        for (int i = 0; i < K_; ++i) if (int(sel[i]) == slot[j]) {{ selj = true; break; }}
        if (!selj) {{ vic = j; break; }}
    }}
    bool doswap = (nom < E_) && (vic < R_);
    int vicc = vic < R_ ? vic : R_ - 1;
    src[0] = uint(doswap ? nom : slot[vicc]);
    for (int j = 0; j < R_; ++j) new_slot[j] = slot[j];
    if (doswap) new_slot[vicc] = nom;
    int slot0 = new_slot[0];
    for (int i = 0; i < K_; ++i) {{
        int s = int(sel[i]);
        bool now_res = (loc[i] >= 0) || (doswap && s == nom);
        eff[i] = uint(now_res ? s : slot0);
    }}
    """
    kern = mx.fast.metal_kernel(name=f"deploy_step_{E}_{R}_{K}",
                                input_names=["slot", "sel"],
                                output_names=["new_slot", "src", "eff"],
                                source=src_code)

    def call(slot, sel):
        return kern(inputs=[slot, sel], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                    output_shapes=[(R,), (1,), (K,)],
                    output_dtypes=[mx.int32, mx.uint32, mx.uint32])

    return call


@lru_cache(maxsize=None)
def _floor_src_step(N):
    """RAM floor: cycled source ids, data-dependent on this layer's routing
    (+ sel[0]*0 -- fetch-on-miss causality, prevents Metal hoisting the copies
    into earlier layers). Compiled so the whole thing is 1-2 tiny kernels."""
    ar = mx.arange(N, dtype=mx.int32)

    def f(sel, st0):
        return (ar + st0 + sel[0].astype(mx.int32) * 0).astype(mx.uint32)

    return mx.compile(f)


def _pin_thread_qos():
    """Pin the calling thread to QOS_CLASS_USER_INTERACTIVE (macOS).

    Measured harness fix, not a semantic change: whenever the decode thread
    BLOCKS (mx.eval wait, fetch wait -- and a plain time.sleep reproduces it),
    the macOS scheduler demotes/migrates it, and the next ~ms of graph
    encoding and GPU-completion waits run 2-3x slower. A 350 us busy-wait gap
    is perfectly additive; a 350 us blocking sleep costs ~3x the sync-loop
    tax. Pinning QoS keeps post-wake work at full speed. Applied to the bench
    decode thread (all temporal modes equally) and the pread workers."""
    try:
        import ctypes
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.pthread_set_qos_class_self_np(0x21, 0)  # QOS_CLASS_USER_INTERACTIVE
    except Exception:
        pass  # non-macOS or restricted environment: no-op


class _Done:
    """Completion handle for one batch of preads (single event, no per-read
    Future churn)."""
    __slots__ = ("n", "ev", "lock", "err")

    def __init__(self, n):
        import threading
        self.n = n
        self.err = None
        self.ev = threading.Event()
        self.lock = threading.Lock()

    def wait(self):
        self.ev.wait()
        if self.err is not None:
            raise self.err


class _PreadPool:
    """Persistent pread worker threads with a SimpleQueue submission path.
    Replaces ThreadPoolExecutor: same read shapes / same parallelism, but no
    per-call future allocation and one completion event per batch instead of
    one wait per sub-read."""

    def __init__(self, nthreads):
        import queue
        import threading
        self._q = queue.SimpleQueue()
        self._threads = []
        for _ in range(nthreads):
            t = threading.Thread(target=self._run, daemon=True)
            t.start()
            self._threads.append(t)

    def _run(self):
        _pin_thread_qos()
        q = self._q
        while True:
            item = q.get()
            if item is None:
                return
            fd, off, buf, at, ln, done = item
            try:
                got = os.preadv(fd, [memoryview(buf)[at:at + ln]], off)
                if got != ln:
                    done.err = AssertionError(
                        f"short preadv at offset {off}: {got} != {ln}")
            except BaseException as e:  # surfaced at done.wait()
                done.err = e
            with done.lock:
                done.n -= 1
                if done.n == 0:
                    done.ev.set()

    def issue(self, fd, buf, parts):
        """parts: [(file_offset, buf_offset, length)]. Returns a _Done."""
        done = _Done(len(parts))
        put = self._q.put
        for off, at, ln in parts:
            put((fd, off, buf, at, ln, done))
        return done

    def shutdown(self):
        for _ in self._threads:
            self._q.put(None)


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
        # noswap: the table is static, so the id-remap (resident -> itself,
        # non-resident -> slot 0's occupant; the fork's loc[]-derived rule) is
        # a precomputed expert->effective-id lookup -- one gather per layer.
        if self.ctrl.mode == "noswap":
            le = np.full(E, self.slot_of[0], dtype=np.uint32)
            le[self.slot_of] = self.slot_of
            self.loc_eff = mx.array(le)
            mx.eval(self.loc_eff)
        # floor: the timing-emulation GEMM reads all R slots; the index vector
        # is a per-layer constant -- materialize once instead of per token.
        if self.ctrl.mode == "floor":
            self._eff_all = mx.arange(self.R, dtype=mx.uint32).reshape(1, 1, self.R)
            mx.eval(self._eff_all)
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

    def _issue_disk(self, n):
        """Non-blocking issue of n experts' preads from the >RAM disk pool into
        a preallocated numpy ring buffer, via the persistent worker pool.
        Returns (_Done, bytes). Offset streams are byte-identical to the
        original blocking implementation: n==1 (deploy / floor_n=1) splits the
        single expert into TEMPORAL_DISK_SPLIT parallel sub-reads and advances
        the cycle by 7919; n>=2 (floor) reads n whole experts at cycled
        offsets (stride 7919) and advances by n*7919+1. Causality is enforced
        by the callers' control flow: issue only after this layer's routing /
        residency decision has been evaluated, wait before the expert GEMM is
        submitted for execution."""
        ctrl = self.ctrl
        eb, E_d, L = self.expert_bytes, ctrl.disk_E, self.disk_idx
        buf = ctrl.disk_ring[ctrl.disk_ring_i]
        ctrl.disk_ring_i = (ctrl.disk_ring_i + 1) % len(ctrl.disk_ring)
        if n == 1:
            off = (L * E_d + (self._dcycle % E_d)) * eb
            self._dcycle = (self._dcycle + 7919) % E_d
            parts = [(off + a, a, ln) for a, ln in ctrl.disk_subparts]
        else:
            parts = [((L * E_d + (self._dcycle + i * 7919) % E_d) * eb,
                      i * eb, eb) for i in range(n)]
            self._dcycle = (self._dcycle + n * 7919 + 1) % E_d
        return ctrl.disk_pool.issue(ctrl.disk_fd, buf, parts), eb * n

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

        # device modes: zero host syncs on the RAM tier, virtual slots
        # (effective EXPERT ids over the cold pool -- op-order-identical to the
        # reference emulator), tier-crossing bytes as staged copies ordered
        # before the GEMM.
        sel = inds.reshape(self.k)
        disk_tick = None                      # (causality_arr, n_fetch)
        if mode == "deploy":
            if self.ctrl.diag == "residcpu":   # experiment: tiny residency ops on CPU stream
                with mx.stream(mx.cpu):
                    self.slot_mx, src, eff = _deploy_step(
                        self.E, self.R)(self.slot_mx, sel)
                src = src.astype(mx.uint32)
                eff = eff.astype(mx.uint32)
            else:
                self.slot_mx, src, eff = _deploy_step_fast(
                    self.E, self.R, self.k)(self.slot_mx, sel)
            if self.ctrl.disk_fd is not None:
                # causality token = src (swap decision before fetch issue)
                self.ctrl._disk_tick_pre()
                disk_tick = (src, 1)
            else:
                x, nb = self._stage(x, src, 1)
                self.ctrl.copied_bytes += nb
            eff = eff.reshape(1, 1, self.k)
        elif mode == "noswap":
            # machinery on, zero swaps: id-remap through the static residency
            # table, precomputed at reset as an expert->effective-id lookup
            # (the fork's loc[]-derived remap) -- one gather kernel per layer.
            eff = mx.take(self.loc_eff, sel).reshape(1, 1, self.k)
        elif mode == "floor" and self.ctrl.disk_fd is not None:
            # causality token = sel (routing first; also the n0 sync-loop
            # baseline). The deferred two-phase tick (see _disk_tick_pre/post)
            # keeps causality identical to the old per-layer-mx.eval loop while
            # removing pure harness serialization.
            self.ctrl._disk_tick_pre()
            disk_tick = (sel, self.ctrl.N)
            eff = self._eff_all               # timing emulation: reads all R slots
        elif mode == "floor":
            N, E = self.ctrl.N, self.E
            if N > 0:
                st0 = 0 if self.ctrl.same_window else self._cycle % max(1, E - N)
                self._cycle += N
                # cycled sources, made DATA-DEPENDENT on this layer's router
                # output (+ sel[0]*0): fetch-on-miss causality. Without this the
                # indices are graph constants and Metal hoists the copies into
                # earlier layers' compute = illegal prefetch for a floor.
                srcs = _floor_src_step(N)(sel, mx.array(st0, dtype=mx.int32))
                x, nb = self._stage(x, srcs, N)
                self.ctrl.copied_bytes += nb
            # timing emulation: GEMM shape/cost identical for any R ids
            eff = self._eff_all
        else:
            raise ValueError(f"unknown temporal mode {mode!r}")
        y = self.block.switch_mlp(x, eff)     # built while prev fetch flies
        out = (y * scores[..., None]).sum(axis=-2)
        if disk_tick is not None:
            self.ctrl._disk_tick_post(disk_tick[0], self, disk_tick[1])
            if self.disk_idx == self.ctrl.n_last:
                self.ctrl._flush_pending()    # last layer: fetch lands before
        return out                            #   the head/GEMM can be submitted

    # ---- ROUTER-EARLY split forward (disk deploy only; driven by the decoder
    # block in model.py so it can interleave attention between issue and wait).
    # The router + residency decision + expert-fetch ISSUE run on the
    # PRE-attention input; attention runs while the pread threads are in flight;
    # the expert GEMM runs POST-attention on the post-attention hidden state but
    # with this pre-attention routing decision. Because the disk bytes never
    # feed the math (the GEMM reads the cold pool through effective ids, exactly
    # as the sync deploy device path does), overlap changes only TIMING -- the
    # logits are identical to the TEMPORAL_EARLY_NOOVERLAP control bit-for-bit.
    def route_submit(self, r_in):
        """Step 1: route on the pre-attention (post_attention_layernorm'd)
        residual, run the branchless deploy residency step, SUBMIT the graph
        delta (which contains the previous layer's expert GEMM -- so the
        previous fetch is waited first). Returns (src, eff, scores); the
        caller builds the attention graph while the GPU executes this delta,
        then calls issue_after_route.

        Deferred-wait structure (harness-overhead removal, same causality):
        every expert GEMM still executes only after its fetch landed, and the
        fetch still issues only after this layer's swap decision has been
        computed; what overlaps now is python graph building (a real engine's
        CPU work) with GPU execution and with the pread flight."""
        ctrl = self.ctrl
        inds, scores = self.block.route(r_in)
        sel = inds.reshape(self.k)
        # same branchless self-copy drive as the sync deploy path
        self.slot_mx, src, eff = _deploy_step_fast(
            self.E, self.R, self.k)(self.slot_mx, sel)
        ctrl._early_wait()                # prev layer's fetch must land before
        mx.async_eval(src)                #   its GEMM (in this delta) submits
        return src, eff, scores

    def issue_after_route(self, src):
        """Steps 2-3: wait the swap decision (computed while attention was
        being BUILT), then issue the overlapped single-expert fetch.
        TEMPORAL_EARLY_NOOVERLAP=1 blocks the fetch here (sequential
        overlap-isolation control, unchanged)."""
        ctrl = self.ctrl
        mx.eval(src)                      # swap decision computed -> fetch may issue
        if ctrl.diag == "nocopy":
            return
        done, nb = self._issue_disk(1)
        if ctrl.early_nooverlap:          # overlap-isolation control: block now
            done.wait()
            ctrl.copied_bytes += nb
        else:
            ctrl._early_pending = (done, nb)

    def expert_finish(self, h_norm, eff, scores):
        """Steps 5-6: build the expert GLU on the POST-attention hidden state
        with the pre-attention routing decision. The in-flight fetch is waited
        by the NEXT layer's route_submit (or here on the last layer) -- always
        before the GEMM is submitted for execution."""
        y = self.block.switch_mlp(h_norm, eff.reshape(1, 1, self.k))
        out = (y * scores[..., None]).sum(axis=-2)
        if self.disk_idx == self.ctrl.n_last:
            self.ctrl._early_wait()       # last layer: fetch lands before the
        return out                        #   head can submit this GEMM


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
        self.disk_pool = None
        self.mode = mode
        self.N = N
        _pin_thread_qos()  # decode-thread scheduling fix (see docstring)
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
        self._pending = None          # deferred (causality_arr, layer, n) for
        self._pending_done = None     #   the sync loop (two-phase tick) /
        self._early_pending = None    #   router-early in-flight fetch
        self.n_last = len(self.layers) - 1
        if self.disk_path and mode in ("deploy", "floor"):
            import fcntl
            self.disk_fd = os.open(self.disk_path, os.O_RDONLY)
            fcntl.fcntl(self.disk_fd, 48, 1)   # F_NOCACHE
            fsz = os.fstat(self.disk_fd).st_size
            eb = self.layers[0].expert_bytes
            stride = len(self.layers) * eb
            assert fsz % stride == 0, (
                f"disk pool {self.disk_path!r} size {fsz} not a multiple of "
                f"n_layers*expert_bytes ({len(self.layers)}*{eb}={stride})")
            self.disk_E = fsz // stride
            self.disk_pool = _PreadPool(self.disk_qd)
            self.disk_split = int(os.environ.get("TEMPORAL_DISK_SPLIT", "8"))
            # precomputed (buf_offset, length) sub-parts of a single-expert
            # split read (same split-8 shape as before)
            step = -(-eb // self.disk_split)
            self.disk_subparts = [(j * step, min(step, eb - j * step))
                                  for j in range(self.disk_split)]
            ring_sz = max(1, N) * eb
            self.disk_ring = [np.empty(ring_sz, dtype=np.uint8) for _ in range(2)]
            self.disk_ring_i = 0
            for i, tl in enumerate(self.layers):
                tl.disk_idx = i
                tl._dcycle = 0

    # ---- regime-2 sync-loop structure (floor + sync deploy disk rows) ----
    # Deferred-wait per-layer tick: layer L's fetch-on-miss sequence
    # (wait routing_L computed -> issue fetch_L -> wait fetch_L) runs at layer
    # L+1, just BEFORE the graph delta containing layer L's expert GEMM is
    # submitted. Causality is exactly the old per-layer-mx.eval loop's:
    # fetches issue only after their layer's routing/decision has executed,
    # and every expert GEMM is submitted only after its fetch completed, with
    # the GPU idle during the fetch itself. What changes is pure harness
    # overhead: python graph building of the next layer now overlaps the GPU's
    # execution of the submitted delta instead of serializing after it.
    def _disk_tick_pre(self):
        """First half of the deferred tick: wait the previous layer's routing,
        ISSUE its fetch (nonblocking). The caller then builds this layer's
        expert-GEMM graph while the fetch is in flight (python-only work, a
        real engine's CPU side), and calls _disk_tick_post."""
        p = self._pending
        if p is None:
            return
        self._pending = None
        causal, layer, n = p
        mx.eval(causal)                   # wait: routing/decision computed
        if n and self.diag != "nocopy":
            self._pending_done = layer._issue_disk(n)

    def _disk_tick_post(self, causal, layer, n_fetch):
        """Second half: wait the in-flight fetch (GPU idle -- the sync row's
        defining cost), then submit this layer's graph delta (which contains
        the PREVIOUS layer's expert GEMM -- hence the wait-before-submit)."""
        d = self._pending_done
        if d is not None:
            self._pending_done = None
            done, nb = d
            done.wait()
            self.copied_bytes += nb
        mx.async_eval(causal)             # submit delta (incl. prev layer GEMM)
        self._pending = (causal, layer, n_fetch)

    def _flush_pending(self):
        self._disk_tick_pre()
        d = self._pending_done
        if d is not None:
            self._pending_done = None
            done, nb = d
            done.wait()
            self.copied_bytes += nb

    def _early_wait(self):
        p = self._early_pending
        if p is None:
            return
        self._early_pending = None
        done, nb = p
        done.wait()
        self.copied_bytes += nb

    @property
    def n_moe_layers(self):
        return len(self.layers)

    def reset(self):
        self.copied_bytes = 0
        self.retained.clear()
        self.retained_b = 0
        self._pending = None
        self._pending_done = None
        self._early_pending = None
        for tl in self.layers:
            tl.reset()

    def disable(self):
        for layer in self._model.model.layers:
            layer.mlp.temporal = None
        if self.disk_pool is not None:
            self.disk_pool.shutdown()
            self.disk_pool = None
        if self.disk_fd is not None:
            try:
                os.close(self.disk_fd)
            except OSError:
                pass
            self.disk_fd = None
