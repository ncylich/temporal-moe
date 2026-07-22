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
def _deploy_step_masked(E, R, K):
    """_deploy_step_fast plus the fork's TEMPORAL_UNIFIED_OVERLAP masking
    order: also emits `perm` [K] -- selection positions in stable order with
    the position of the FETCHED expert (nominee if a swap fires, else the
    victim slot's occupant) moved LAST iff that expert is among the selected.
    The caller runs the first K-1 (resident-hit) contributions while the
    fetch is in flight and only the last contribution behind the fetch.
    ids_sorted[j] = eff[perm[j]]. Residency decisions identical to
    _deploy_step / _deploy_step_fast."""
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
    int f = doswap ? nom : slot[vicc];
    src[0] = uint(f);
    for (int j = 0; j < R_; ++j) new_slot[j] = slot[j];
    if (doswap) new_slot[vicc] = nom;
    int slot0 = new_slot[0];
    int effv[K_];
    for (int i = 0; i < K_; ++i) {{
        int s = int(sel[i]);
        bool now_res = (loc[i] >= 0) || (doswap && s == nom);
        effv[i] = now_res ? s : slot0;
    }}
    int m = -1;
    for (int i = 0; i < K_; ++i) if (int(sel[i]) == f) {{ m = i; break; }}
    int w = 0;
    for (int i = 0; i < K_; ++i) if (i != m) perm[w++] = uint(i);
    if (m >= 0) perm[K_ - 1] = uint(m);
    for (int j = 0; j < K_; ++j) ids_sorted[j] = uint(effv[int(perm[j])]);
    """
    kern = mx.fast.metal_kernel(name=f"deploy_step_masked_{E}_{R}_{K}",
                                input_names=["slot", "sel"],
                                output_names=["new_slot", "src", "ids_sorted",
                                              "perm"],
                                source=src_code)

    def call(slot, sel):
        return kern(inputs=[slot, sel], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                    output_shapes=[(R,), (1,), (K,), (K,)],
                    output_dtypes=[mx.int32, mx.uint32, mx.uint32, mx.uint32])

    return call


@lru_cache(maxsize=None)
def _deploy_copy_step(E, R, K, EB):
    """RAM-tier fused residency + swap-copy kernel: ONE launch computes the
    k_residency decisions (bit-identical logic to _deploy_step_fast) AND
    copies the fetched expert's 663,552B row cold_flat[src] into a fresh
    staging buffer at full bandwidth (each threadgroup recomputes the tiny
    decision locally -- no cross-threadgroup sync needed -- then copies its
    16B-chunk slice). Replaces [step kernel + take gather + astype plumbing]:
    ~2 launches and their gaps collapse into ~1 launch + ~7us of bytes.
    Branchless p=1.0 drive preserved: no-nominee layers self-copy the victim
    occupant's row (real bytes, no-op math), exactly as before."""
    assert EB % 16 == 0
    CHUNKS = EB // 16
    src_code = f"""
    const int E_ = {E}; const int R_ = {R}; const int K_ = {K};
    const uint CHUNKS_ = {CHUNKS};
    threadgroup int tg_src;
    if (thread_index_in_threadgroup == 0) {{
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
        int f = doswap ? nom : slot[vicc];
        tg_src = f;
        if (thread_position_in_grid.x == 0) {{
            src[0] = uint(f);
            for (int j = 0; j < R_; ++j) new_slot[j] = slot[j];
            if (doswap) new_slot[vicc] = nom;
            int slot0 = doswap && vicc == 0 ? nom : slot[0];
            for (int i = 0; i < K_; ++i) {{
                int s = int(sel[i]);
                bool now_res = (loc[i] >= 0) || (doswap && s == nom);
                eff[i] = uint(now_res ? s : slot0);
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uint4* srcp =
        (const device uint4*)(cold + (ulong)tg_src * {EB});
    device uint4* dstp = (device uint4*)staged;
    for (uint c = thread_position_in_grid.x; c < CHUNKS_;
         c += threads_per_grid.x) {{
        dstp[c] = srcp[c];
    }}
    """
    kern = mx.fast.metal_kernel(name=f"deploy_copy_{E}_{R}_{K}_{EB}",
                                input_names=["slot", "sel", "cold"],
                                output_names=["new_slot", "src", "eff",
                                              "staged"],
                                source=src_code)
    tpg = 1024
    grid = min(((CHUNKS + 3) // 4 + tpg - 1) // tpg * tpg, CHUNKS)
    grid = ((grid + tpg - 1) // tpg) * tpg   # 4 chunks/thread grid-stride

    def call(slot, sel, cold):
        return kern(inputs=[slot, sel, cold], grid=(grid, 1, 1),
                    threadgroup=(tpg, 1, 1),
                    output_shapes=[(R,), (1,), (K,), (EB // 4,)],
                    output_dtypes=[mx.int32, mx.uint32, mx.uint32, mx.uint32])

    return call


@lru_cache(maxsize=None)
def _floor_copy_step(N, EB):
    """RAM-floor fused copy kernel: N cycled-source expert rows (sources
    st0+i, the existing window formula) copied cold->staging in ONE launch.
    Taking `sel` as an input preserves fetch-on-miss causality in-graph (the
    copy is ordered after this layer's routing; no hoisting)."""
    assert EB % 16 == 0
    CHUNKS = EB // 16
    src_code = f"""
    const uint CHUNKS_ = {CHUNKS};
    uint c = thread_position_in_grid.x;
    uint dummy = sel[0];               // causality: data-dependence on routing
    if (c < CHUNKS_ * {N}) {{
        uint row = c / CHUNKS_;
        uint off = c % CHUNKS_;
        uint srcrow = st0[0] + row + (dummy & 0u);
        const device uint4* srcp =
            (const device uint4*)(cold + (ulong)srcrow * {EB});
        ((device uint4*)staged)[c] = srcp[off];
    }}
    """
    kern = mx.fast.metal_kernel(name=f"floor_copy_{N}_{EB}",
                                input_names=["sel", "st0", "cold"],
                                output_names=["staged"],
                                source=src_code)
    tpg = 256
    total = CHUNKS * N
    grid = ((total + tpg - 1) // tpg) * tpg

    def call(sel, st0, cold):
        return kern(inputs=[sel, st0, cold], grid=(grid, 1, 1),
                    threadgroup=(tpg, 1, 1),
                    output_shapes=[(N * EB // 4,)],
                    output_dtypes=[mx.uint32])

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


def _respin(ms=100):
    """Post-sleep P-core re-promotion spin (macOS, measured): waking from a
    multi-second sleep lands the decode thread on an E-core roughly 1 rep in
    8, and the ENTIRE following rep then runs there (~2.2x on-CPU time for
    identical work; QoS stays USER_INTERACTIVE throughout, so QoS pinning
    alone does not prevent it). A short untimed CPU burst right after the
    cooldown sleep promotes the thread back to a P-core before the timed
    block starts. Runs strictly OUTSIDE timed regions."""
    import time
    t0 = time.perf_counter()
    x = 1.0
    while (time.perf_counter() - t0) < ms / 1e3:
        x = x * 1.0000001 + 1e-9
    return x


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
        # Fixed-shape subgraphs compiled once per layer (decode x is always
        # [1,1,H]; no KV-cache shapes involved, so no retracing): the router
        # and the expert-GLU+weighted-sum. Cuts per-layer graph-encode cost
        # (MLX encodes per op; ~10 ops -> 1 compiled node each) on the
        # latency-critical disk paths. Bitwise-identical math (exactness gates
        # verify against the uncompiled reference emulator).
        if self.ctrl.mode in ("deploy", "noswap", "floor"):
            self._route_c = mx.compile(self.block.route)
            sm = self.block.switch_mlp
            self._glu_c = mx.compile(
                lambda x, eff, scores:
                    (sm(x, eff) * scores[..., None]).sum(axis=-2))
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
            evs = [self._eff_all]
            N = self.ctrl.N
            if 0 < N < self.R:                # masked split slices (disk rows)
                self._eff_hit = self._eff_all[..., :self.R - N]
                self._eff_miss = self._eff_all[..., self.R - N:]
                evs += [self._eff_hit, self._eff_miss]
            mx.eval(*evs)
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

    def _issue_direct(self):
        """Single-expert fetch as ONE synchronous preadv on the calling
        thread (sync_direct mode). Byte- and offset-identical to the pool's
        n==1 path at TEMPORAL_DISK_SPLIT=1; the GIL is released for the
        duration of the syscall. Returns (None, bytes)."""
        ctrl = self.ctrl
        eb, E_d, L = self.expert_bytes, ctrl.disk_E, self.disk_idx
        buf = ctrl.disk_ring[ctrl.disk_ring_i]
        ctrl.disk_ring_i = (ctrl.disk_ring_i + 1) % len(ctrl.disk_ring)
        off = (L * E_d + (self._dcycle % E_d)) * eb
        self._dcycle = (self._dcycle + 7919) % E_d
        got = os.preadv(ctrl.disk_fd, [memoryview(buf)[0:eb]], off)
        assert got == eb, f"short preadv at offset {off}: {got} != {eb}"
        return None, eb

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

    def _retain_and_order(self, x, staged, nbytes):
        """Retention (>SLC write-target rotation over one token's span --
        same deque semantics as _stage) + order the copy BEFORE this layer's
        expert GEMM. Returns x with the dependency edge."""
        ctrl = self.ctrl
        if ctrl.retain_bytes:
            ctrl.retained.append(staged)
            ctrl.retained_b += staged.nbytes
            while ctrl.retained_b > ctrl.retain_bytes:
                ctrl.retained_b -= ctrl.retained.popleft().nbytes
        ctrl.copied_bytes += nbytes
        return mx.depends([x], [staged])[0]

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
        mode = self.ctrl.mode
        if mode in ("deploy_ref", "lazy_full"):
            inds, scores = self.block.route(x)      # gate modes: uncompiled
        else:
            inds, scores = self._route_c(x)         # true top-k routing (unchanged
            #   math; compiled for encode cost -- gates check bitwise vs reference)

        if mode in ("deploy_ref", "lazy_full"):
            # host-side gate modes: sync on the selection (slow; correctness only)
            sel_np = np.array(inds[0, 0]).astype(np.int64)
            selset = set(sel_np.tolist())
            if mode == "deploy_ref":
                # G2b-ii reference emulator: SAME deploy residency rule, but the
                # expert GEMM is computed straight from the COLD pool via
                # effective ids -- no hot slots, no byte copies.
                # ctrl.split_order: mirror the masked (fork
                # TEMPORAL_UNIFIED_OVERLAP-order) structure of the bench deploy
                # path -- fetched-expert contribution moved last and summed
                # separately ((k-1)-sum + 1), so the float reduction order
                # matches the split GEMM bit-for-bit.
                if self.ctrl.split_order:
                    loc, slot, R = self.loc, self.slot_of, self.R
                    nominee = -1
                    for e in range(self.E):
                        if loc[e] < 0 and e in selset:
                            nominee = e
                            break
                    victim = -1
                    for s in range(R):
                        re = slot[s]
                        if re < 0 or re not in selset:
                            victim = s
                            break
                    do = nominee >= 0 and victim >= 0
                    vicc = victim if victim >= 0 else R - 1
                    f = nominee if do else int(slot[vicc])
                    self._resid_deploy(selset)
                    eff = np.where(self.loc[sel_np] >= 0, sel_np,
                                   self.slot_of[0])
                    m = -1
                    for i in range(self.k):
                        if int(sel_np[i]) == f:
                            m = i
                            break
                    perm = [i for i in range(self.k) if i != m]
                    if m >= 0:
                        perm.append(m)
                    ids = eff[perm]
                    sc = mx.take_along_axis(
                        scores, mx.array(np.array(perm, dtype=np.uint32)
                                         ).reshape(1, 1, self.k), axis=-1)
                    kh = self.k - 1
                    y1 = self.block.switch_mlp(
                        x, mx.array(ids[:kh].reshape(1, 1, kh)))
                    y2 = self.block.switch_mlp(
                        x, mx.array(ids[kh:].reshape(1, 1, 1)))
                    return ((y1 * sc[..., :kh, None]).sum(axis=-2)
                            + (y2 * sc[..., kh:, None]).sum(axis=-2))
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
            k = self.k
            disk = self.ctrl.disk_fd is not None
            masked = (k > 1 and self.ctrl.diag != "residcpu"
                      and (disk and not self.ctrl.sync_inline
                           and not self.ctrl.no_mask
                           or self.ctrl.split_order))
            if masked:
                # Fork TEMPORAL_UNIFIED_OVERLAP analog (masked split, same
                # token): the k-1 resident-hit expert contributions execute
                # WHILE the fetch is in flight; only the fetched expert's
                # contribution waits on it. The step kernel emits the ids in
                # miss-last order + the permutation for the scores.
                self.slot_mx, src, ids_s, perm = _deploy_step_masked(
                    self.E, self.R, k)(self.slot_mx, sel)
                sc = mx.take_along_axis(scores, perm.reshape(1, 1, k), axis=-1)
                kh = k - 1
                if disk and self.ctrl.ts is not None:
                    # STREAM ENGINE: no python-side sync at all. SignalFetch
                    # commits the CB (handler preads + signals the layer's
                    # shared event); the hit part encodes into the NEXT CB
                    # (overlaps the fetch); WaitFetch gates only the fetched
                    # expert's contribution. Whole token pipelines like the
                    # ceiling path.
                    ts = self.ctrl.ts
                    self._wait_val += 1
                    sig = ts.signal_fetch(src, self.disk_idx)
                    hit = self._glu_c(mx.depends([x], [sig])[0],
                                      ids_s[:kh].reshape(1, 1, kh), sc[..., :kh])
                    # NOTE (measured): Metal defers starting a CB whose stream
                    # holds an unsatisfied event wait, so these hits do NOT
                    # overlap the fetch (0/3599 in the trace control). Forcing
                    # them into their own CB via ts.commit_boundary(hit) makes
                    # them overlap but costs an extra CB boundary -- a wash at
                    # B=1 (~40us window vs ~50us boundary). Left un-split.
                    gate = ts.wait_fetch(hit, sig, self.disk_idx, self._wait_val)
                    return gate + self._glu_c(
                        mx.depends([x], [gate])[0],
                        ids_s[kh:].reshape(1, 1, 1), sc[..., kh:])
                if disk:
                    self.ctrl._disk_tick_pre()
                    hit = self._glu_c(x, ids_s[:kh].reshape(1, 1, kh),
                                      sc[..., :kh])
                    # submit hit-GEMM delta; issuer waits src then preads --
                    # the fetch overlaps the hit-GEMM execution
                    self.ctrl._disk_tick_post(hit, self, src, 1)
                    out = hit + self._glu_c(x, ids_s[kh:].reshape(1, 1, 1),
                                            sc[..., kh:])
                    if self.disk_idx == self.ctrl.n_last:
                        self.ctrl._flush_pending()
                else:
                    # RAM tier, same op order (copies stay graph-ordered)
                    x, nb = self._stage(x, src, 1)
                    self.ctrl.copied_bytes += nb
                    out = (self._glu_c(x, ids_s[:kh].reshape(1, 1, kh),
                                       sc[..., :kh])
                           + self._glu_c(x, ids_s[kh:].reshape(1, 1, 1),
                                         sc[..., kh:]))
                return out
            if not disk and self.ctrl.diag == "" and k > 1 and self.ctrl.ram_fused:
                # RAM tier, fused engine: ONE kernel computes the residency
                # decisions AND copies the fetched expert's row cold->staging
                # at full bandwidth (branchless p=1.0 drive in-kernel).
                self.slot_mx, src, eff, staged = _deploy_copy_step(
                    self.E, self.R, k, self.expert_bytes)(
                        self.slot_mx, sel, self.cold_flat)
                x = self._retain_and_order(x, staged, self.expert_bytes)
                eff = eff.reshape(1, 1, k)
            else:
                if self.ctrl.diag == "residcpu":   # experiment: tiny residency ops on CPU stream
                    with mx.stream(mx.cpu):
                        self.slot_mx, src, eff = _deploy_step(
                            self.E, self.R)(self.slot_mx, sel)
                    src = src.astype(mx.uint32)
                    eff = eff.astype(mx.uint32)
                else:
                    self.slot_mx, src, eff = _deploy_step_fast(
                        self.E, self.R, self.k)(self.slot_mx, sel)
                if disk:
                    # causality token = src (swap decision before fetch issue)
                    if self.ctrl.sync_inline:
                        self.ctrl._disk_tick_inline(src, self, 1)
                    else:
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
            N, R = self.ctrl.N, self.R
            if self.ctrl.ts is not None:
                # STREAM ENGINE (identical machinery; fairness symmetry): the
                # R-N hit-slot contributions overlap the N-expert fetch; the
                # N miss slots (or, for N==0/N>=R, the whole GEMM) are gated
                # on the layer's shared event.
                ts = self.ctrl.ts
                self._wait_val += 1
                sig = ts.signal_fetch(sel, self.disk_idx)
                if 0 < N < R:
                    hits = R - N
                    hit = self._glu_c(mx.depends([x], [sig])[0],
                                      self._eff_hit, scores[..., :hits])
                    gate = ts.wait_fetch(hit, sig, self.disk_idx, self._wait_val)
                    return gate + self._glu_c(
                        mx.depends([x], [gate])[0],
                        self._eff_miss, scores[..., hits:])
                gate = ts.wait_fetch(x, sig, self.disk_idx, self._wait_val)
                return self._glu_c(gate, self._eff_all, scores)
            if self.ctrl.sync_inline:
                self.ctrl._disk_tick_inline(sel, self, N)
            elif 0 < N < R and not self.ctrl.no_mask:
                # same masked split as deploy (fairness symmetry): the R-N
                # hit-slot contributions execute while the N-expert fetch is
                # in flight; the N miss-slot contributions wait on it.
                self.ctrl._disk_tick_pre()
                hits = R - N
                hit = self._glu_c(x, self._eff_hit, scores[..., :hits])
                self.ctrl._disk_tick_post(hit, self, sel, N)
                out = hit + self._glu_c(x, self._eff_miss, scores[..., hits:])
                if self.disk_idx == self.ctrl.n_last:
                    self.ctrl._flush_pending()
                return out
            else:
                self.ctrl._disk_tick_pre()
                disk_tick = (sel, N)
            eff = self._eff_all               # timing emulation: reads all R slots
        elif mode == "floor":
            N, E = self.ctrl.N, self.E
            if N > 0:
                st0 = 0 if self.ctrl.same_window else self._cycle % max(1, E - N)
                self._cycle += N
                if self.ctrl.diag == "" and self.ctrl.ram_fused:
                    # fused engine: N cycled-source rows copied cold->staging
                    # in ONE kernel; taking `sel` as input keeps fetch-on-miss
                    # causality in-graph (no hoisting).
                    staged = _floor_copy_step(N, self.expert_bytes)(
                        sel, mx.array([st0], dtype=mx.uint32), self.cold_flat)[0]
                    x = self._retain_and_order(x, staged, N * self.expert_bytes)
                else:
                    # diag paths (nocopy etc.): original staged-take form.
                    # cycled sources, DATA-DEPENDENT on this layer's router
                    # output (+ sel[0]*0): fetch-on-miss causality.
                    srcs = _floor_src_step(N)(sel, mx.array(st0, dtype=mx.int32))
                    x, nb = self._stage(x, srcs, N)
                    self.ctrl.copied_bytes += nb
            # timing emulation: GEMM shape/cost identical for any R ids
            eff = self._eff_all
        else:
            raise ValueError(f"unknown temporal mode {mode!r}")
        out = self._glu_c(x, eff, scores)     # built while prev fetch flies
        if disk_tick is not None:
            self.ctrl._disk_tick_post(disk_tick[0], self, disk_tick[0],
                                      disk_tick[1])
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
        inds, scores = self._route_c(r_in)
        sel = inds.reshape(self.k)
        # same branchless self-copy drive as the sync deploy path
        self.slot_mx, src, eff = _deploy_step_fast(
            self.E, self.R, self.k)(self.slot_mx, sel)
        ctrl._early_wait()                # prev layer's fetch must land before
        mx.async_eval(src)                #   its GEMM (in this delta) submits
        return src, eff, scores

    def issue_after_route(self, src):
        """Steps 2-3: hand (wait swap decision -> issue single-expert fetch)
        to the issuer thread; the decode thread proceeds straight to
        submitting attention, which the pread then overlaps. The decision is
        still computed before the fetch issues (issuer's eval), and the fetch
        is waited before this layer's expert GEMM can be submitted.
        TEMPORAL_EARLY_NOOVERLAP=1 keeps the fully sequential inline control:
        wait decision, issue, block -- no issuer, no overlap."""
        ctrl = self.ctrl
        if ctrl.early_nooverlap:          # overlap-isolation control
            mx.eval(src)                  # swap decision computed
            if ctrl.diag == "nocopy":
                return
            done, nb = self._issue_disk(1)
            done.wait()
            ctrl.copied_bytes += nb
            return
        n = 0 if ctrl.diag == "nocopy" else 1
        if ctrl.early_inline:
            # inline issue: the decode thread waits the swap decision itself,
            # issues, and defers only the fetch-completion wait (the pread
            # then overlaps attention submission + graph building).
            mx.eval(src)
            ctrl._early_pending = self._issue_disk(1) if n else None
        else:
            ctrl._early_pending = ctrl._hand_to_issuer(src, self, n)

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
        self.early_inline = os.environ.get("TEMPORAL_EARLY_INLINE", "1") == "1"  # inline wins: issuer pays GIL contention
        # sync-tick structure A/B: issuer (default) vs the pre-gen-2 fully
        # inline eval+fetch (diagnostic / fallback)
        self.sync_inline = os.environ.get("TEMPORAL_SYNC_INLINE", "") == "1"
        # masked (fork TEMPORAL_UNIFIED_OVERLAP-order) split on the RAM deploy
        # path + reference emulator: used by the G2 gate and the one-off RAM
        # measurement; the DISK deploy/floor rows use the masked order always
        # unless TEMPORAL_NO_MASK=1 (the same-session mask A/B control).
        self.split_order = os.environ.get("TEMPORAL_SPLIT_ORDER", "") == "1"
        self.no_mask = os.environ.get("TEMPORAL_NO_MASK", "") == "1"
        # TEMPORAL_SYNC_DIRECT=1: single-expert fetches run as a direct
        # synchronous pread on the decode thread inside the deferred tick (no
        # issuer thread, no GIL handoff hops); n>=2 fetches keep the pool
        # (they need parallel reads). GIL switch interval knob for the
        # contention experiment.
        self.sync_direct = os.environ.get("TEMPORAL_SYNC_DIRECT", "") == "1"
        # TEMPORAL_RAM_FUSED=0: fall back to the unfused RAM engine (step
        # kernel + staged take) for A/B; default = fused single-kernel engine.
        self.ram_fused = os.environ.get("TEMPORAL_RAM_FUSED", "1") == "1"
        # TEMPORAL_STREAM=1: engine 3 -- the whole token submits as ONE
        # pipelined graph (like the ceiling); per MoE layer a SignalFetch
        # primitive commits the command buffer with a C++ completed-handler
        # that preads the layer's expert(s) (offset streams byte-identical to
        # _issue_disk) and signals a per-layer MTLSharedEvent, and a WaitFetch
        # primitive encodes a buffer-level wait so only the fetched-expert
        # contribution (masked split unchanged) executes post-fetch. Requires
        # the ext/ extension + the one-line vendored MLX patch
        # (ext/mlx_v0.32.0_temporal.patch). Old issuer engine kept for A/B.
        self.stream_engine = os.environ.get("TEMPORAL_STREAM", "") == "1"
        self.ts = None
        if os.environ.get("TEMPORAL_SWITCHINTERVAL"):
            import sys as _sys
            _sys.setswitchinterval(float(os.environ["TEMPORAL_SWITCHINTERVAL"]))
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
        self._copied_bytes_py = 0
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
        self._pending_box = None      # deferred fetch-completion boxes for the
        self._early_pending = None    #   sync loop / router-early in-flight fetch
        self._issuer_q = None         # issuer thread (started on first use)
        self._issuer_t = None
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
                tl._wait_val = 0
            if self.stream_engine:
                import sys as _sys
                _ext_dir = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "ext")
                if _ext_dir not in _sys.path:
                    _sys.path.insert(0, _ext_dir)
                import _temporal_stream as _ts
                self.ts = _ts
                nfetch = N if mode == "floor" else 1
                # TEMPORAL_STREAM_SIG: "event" = in-stream encodeSignalEvent +
                # pinned service thread (v2); "commit" = per-layer CB commit +
                # completion handler (v1). TEMPORAL_STREAM_TRACE=1 records the
                # hop-by-hop handshake timestamps (mach time) in C++.
                sig_mode = {"commit": 0, "event": 1, "spin": 2, "mtlio": 3}[
                    os.environ.get("TEMPORAL_STREAM_SIG", "commit")]
                _ts.setup(self.disk_path, eb, len(self.layers), nfetch,
                          self.disk_qd, sig_mode,
                          os.environ.get("TEMPORAL_STREAM_TRACE", "") == "1")

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
    # The ISSUER THREAD owns the fetch-on-miss sequencing for one layer at a
    # time: it waits for the layer's routing/decision array to be computed on
    # the GPU (mx.eval from a second thread -- verified safe and GIL-releasing
    # on mlx 0.32), then issues the pread batch, then hands the completion
    # back through a single-item box. The decode thread therefore blocks only
    # ONCE per layer (on fetch completion) instead of twice (GPU wait + fetch
    # wait) -- pure harness-overhead removal. Causality is unchanged: fetches
    # still issue only after their layer's routing/decision has executed
    # (issuer's eval), and the next graph delta -- which contains this layer's
    # expert GEMM -- is still submitted only after the fetch landed (decode
    # waits the box first). For sync rows the GPU stays idle during the fetch
    # exactly as before: nothing further has been submitted at that point.
    def _ensure_issuer(self):
        if self._issuer_q is None:
            import queue
            import threading
            self._issuer_q = queue.SimpleQueue()
            t = threading.Thread(target=self._issuer_run, daemon=True)
            t.start()
            self._issuer_t = t

    def _issuer_run(self):
        _pin_thread_qos()
        # fresh threads have no default GPU stream in MLX; bind explicitly
        # (a second issuer thread in one process otherwise raises
        # "There is no Stream(gpu, 0) in current thread")
        mx.set_default_stream(mx.default_stream(mx.Device(mx.DeviceType.gpu)))
        q = self._issuer_q
        while True:
            item = q.get()
            if item is None:
                return
            causal, layer, n, box = item
            try:
                mx.eval(causal)           # wait: routing/decision computed
                if n and self.diag != "nocopy":
                    box.put(layer._issue_disk(n))
                else:
                    box.put((None, 0))
            except BaseException as e:    # surfaced at the decode-side wait
                box.put(e)

    def _hand_to_issuer(self, causal, layer, n):
        import queue
        self._ensure_issuer()
        box = queue.SimpleQueue()
        self._issuer_q.put((causal, layer, n, box))
        return box

    def _wait_box(self, box):
        r = box.get()                     # issued (or exception)
        if isinstance(r, BaseException):
            raise r
        done, nb = r
        if done is not None:
            done.wait()                   # fetch landed
            self.copied_bytes += nb

    def _disk_tick_pre(self):
        """Wait the PREVIOUS layer's fetch (GPU idle during it -- the sync
        row's defining cost; nothing later has been submitted yet).
        sync_direct mode: perform the whole (wait routing -> fetch) sequence
        right here on the decode thread -- one thread, no GIL handoffs."""
        b = self._pending_box
        if b is None:
            return
        self._pending_box = None
        if isinstance(b, tuple):          # sync_direct deferred (causal, layer, n)
            causal, layer, n = b
            mx.eval(causal)               # wait: routing/decision computed
            if n and self.diag != "nocopy":
                if n == 1:
                    done, nb = layer._issue_direct()
                else:                     # parallel batch still uses the pool
                    done, nb = layer._issue_disk(n)
                if done is not None:
                    done.wait()
                self.copied_bytes += nb
            return
        self._wait_box(b)

    def _disk_tick_inline(self, causal, layer, n):
        """Pre-gen-2 fully inline sync tick (TEMPORAL_SYNC_INLINE=1): submit
        and wait this layer's routing/decision, then fetch, blocking on the
        decode thread. Kept as an A/B fallback."""
        mx.eval(causal)
        if n and self.diag != "nocopy":
            done, nb = layer._issue_disk(n)
            done.wait()
            self.copied_bytes += nb

    def _disk_tick_post(self, submit_arr, layer, causal, n_fetch):
        """Submit this layer's graph delta up to `submit_arr` (contains the
        PREVIOUS layer's expert GEMM -- legal, its fetch was waited in
        tick_pre -- and, on masked rows, this layer's resident-hit GEMM part,
        which the fetch then overlaps) and hand the (wait `causal` computed ->
        issue fetch) sequence to the issuer thread."""
        # submit delta (incl. prev layer GEMM); scheduling `causal` here too
        # guarantees the issuer's eval is a pure wait (it must never have to
        # execute graph nodes itself)
        mx.async_eval(submit_arr, causal)
        if self.sync_direct:
            self._pending_box = (causal, layer, n_fetch)
        else:
            self._pending_box = self._hand_to_issuer(causal, layer, n_fetch)

    def _flush_pending(self):
        self._disk_tick_pre()

    def _early_wait(self):
        p = self._early_pending
        if p is None:
            return
        self._early_pending = None
        if isinstance(p, tuple):          # inline issue: (done, nb)
            done, nb = p
            done.wait()
            self.copied_bytes += nb
        else:
            self._wait_box(p)

    @property
    def copied_bytes(self):
        if self.ts is not None:
            return self.ts.copied_bytes()
        return self._copied_bytes_py

    @copied_bytes.setter
    def copied_bytes(self, v):
        self._copied_bytes_py = v

    @property
    def n_moe_layers(self):
        return len(self.layers)

    def reset(self):
        if self.ts is not None:
            self.ts.reset()
        self.copied_bytes = 0
        self.retained.clear()
        self.retained_b = 0
        self._pending_box = None
        self._early_pending = None
        for tl in self.layers:
            tl.reset()
            tl._wait_val = 0

    def disable(self):
        for layer in self._model.model.layers:
            layer.mlp.temporal = None
        if self.ts is not None:
            self.ts.teardown()
            self.ts = None
        if self._issuer_q is not None:
            self._issuer_q.put(None)
            self._issuer_q = None
        if self.disk_pool is not None:
            self.disk_pool.shutdown()
            self.disk_pool = None
        if self.disk_fd is not None:
            try:
                os.close(self.disk_fd)
            except OSError:
                pass
            self.disk_fd = None
