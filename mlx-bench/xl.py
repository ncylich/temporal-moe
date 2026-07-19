#!/usr/bin/env python3
"""Bigger-than-RAM ("xl") disk-expert decode path for the Mac MLX serving bench.

The xl model (gen_xl_model.py) keeps E=1024 experts/layer as REAL quantized bytes
on disk (experts_flat.bin, 30.6 GB); only R=k=18 experts/layer are hot in RAM.
Each MoE block gets an XLLayer (block.xl) instead of an in-RAM switch_mlp. Fetches
are real preads through a shared F_NOCACHE fd + ThreadPoolExecutor(QD); the fetched
bytes are parsed (gen_xl_model.parse_experts) into q4 tensors and CONSUMED by the
expert GEMM -- unlike temporal.py's timing-only emulation, this is a real data path.

Modes (decode, x=[1,1,H]):
  * noswap    hot-only, static table, ZERO fetches (setup b analog).
  * floor     vanilla-offload floor: fetch N cycled-source experts/layer, build
              compute = concat([hot, staged]) and GEMM over it with indices that
              read every staged expert; persist residency via one take (a real
              engine's slot update). Scores from the true top-k router (timing
              emulation; expert identities are a floor, not the routed ones).
  * deploy    temporal: reuse temporal._deploy_step for the on-device <=1-swap
              residency decision (lowest-index nominee/victim, branchless
              self-copy drive), fetch the 1 nominee expert (QD/2 sub-reads),
              write it into the hot buffer via take over concat([hot, staged]),
              GEMM over the POST-swap hot buffer with the fork's remap-to-slot-0
              rule -> bytewise-equal to temporal.py's deploy_ref.
  * lazy_full GATE ONLY: fetch the whole layer pool and GEMM the true top-k ->
              output exactly equals a full-MoE forward.

Prefill (x=[1,L,H], untimed): hot-only by default (values are decode-latency
irrelevant); exact_prefill=True runs full-pool MoE so a gate's KV cache matches
an in-RAM reference. lazy_full is always full-pool (prefill and decode).
"""
import os
from functools import lru_cache

import mlx.core as mx
import numpy as np

from gen_xl_model import EXPERT_BYTES, PROJ, parse_experts
from temporal import _deploy_step, _noswap_step, expert_glu


@lru_cache(maxsize=None)
def _xl_deploy_map(R):
    """Given (pre-swap slot table, selected experts, post-swap slot table) ->
    (idx_update, row_map). Reuses temporal._deploy_step's identical formulas:
      idx_update[R]: slot ids into concat([hot(R), staged(1)]); the victim slot
        (lowest-index non-selected, same as _deploy_step) reads the staged row R.
      row_map[k]: post-swap slot of each routed expert (resident -> its slot;
        still-non-resident -> slot 0, the fork's remap-to-slot-0)."""
    ridx = mx.arange(R, dtype=mx.int32)

    def f(old_slot, sel, new_slot):
        sel = sel.astype(mx.int32)
        m = sel[:, None] == old_slot[None, :]         # [k,R]
        slot_sel = mx.any(m, axis=0)                  # [R]
        vic = mx.min(mx.where(slot_sel, R, ridx))
        vicc = mx.minimum(vic, R - 1)
        idx_update = mx.where(ridx == vicc, R, ridx)  # victim slot <- staged
        m2 = sel[:, None] == new_slot[None, :]        # [k,R]
        res2 = mx.any(m2, axis=-1)
        slot2 = mx.argmax(m2.astype(mx.int32), axis=-1)
        row_map = mx.where(res2, slot2, 0)
        return idx_update, row_map

    return mx.compile(f)


class XLLayer:
    """Per-MoE-block disk-expert state + forward. Attached as `block.xl`."""

    def __init__(self, block, ctrl, R, disk_idx):
        self.block = block
        self.ctrl = ctrl
        self.E = block.num_experts
        self.k = block.top_k
        self.R = R
        self.disk_idx = disk_idx
        self.gs, self.bits, self.mode_q = ctrl.gs, ctrl.bits, ctrl.qmode
        self._cycle = 0
        self._full_pool = None                         # lazy_full cache
        # initial hot = experts 0..R-1 (kept for reset)
        self._hot0 = ctrl._fetch(disk_idx, list(range(R)))
        self.hot = None
        self.slot_mx = None
        self.reset()

    def reset(self):
        R = self.R
        self.slot_mx = mx.arange(R, dtype=mx.int32)
        self.hot = {name: [mx.array(t) for t in self._hot0[name]] for name in PROJ}
        self._cycle = 0
        mx.eval(self.slot_mx, *[a for t in self.hot.values() for a in t])

    # ---- expert GEMM over a given tensor set (mirrors SwitchGLU op order) ----
    def _glu(self, x, tensors, remap):
        return expert_glu(x, tensors, remap, self.gs, self.bits, self.mode_q)

    # ---- prefill (untimed) ----
    def _prefill(self, x):
        inds, scores = self.block.route(x)                 # [1,L,k]
        if self.ctrl.exact_prefill or self.mode == "lazy_full":
            pool = self._pool()
            y = self._glu(x, pool, inds)
        else:
            # hot-only: static table (slots hold experts 0..R-1) -> slot = eid if
            # resident else 0; values are decode-latency irrelevant.
            slot_map = mx.where(inds < self.R, inds, 0)
            y = self._glu(x, self.hot, slot_map)
        return (y * scores[..., None]).sum(axis=-2)

    def _pool(self):
        if self._full_pool is None:
            self._full_pool = self.ctrl._fetch(self.disk_idx, list(range(self.E)))
            self.ctrl.copied_bytes += self.E * EXPERT_BYTES
        return self._full_pool

    @property
    def mode(self):
        return self.ctrl.mode

    # ---- forward ----
    def __call__(self, x):
        if x.shape[1] != 1:
            return self._prefill(x)
        inds, scores = self.block.route(x)                 # [1,1,k]
        mode = self.mode

        if mode == "lazy_full":
            pool = self._pool()
            y = self._glu(x, pool, inds)
            return (y * scores[..., None]).sum(axis=-2)

        sel = inds.reshape(self.k)

        if mode == "noswap":
            eff = _noswap_step(self.E, self.R)(self.slot_mx, sel)   # eid == slot (static table)
            y = self._glu(x, self.hot, eff.reshape(1, 1, self.k))
            return (y * scores[..., None]).sum(axis=-2)

        if mode == "deploy":
            new_slot, src, _eff = _deploy_step(self.E, self.R)(self.slot_mx, sel)
            idx_update, row_map = _xl_deploy_map(self.R)(self.slot_mx, sel, new_slot)
            mx.eval(src, new_slot, idx_update, row_map)             # causality: decide, then fetch
            staged = self.ctrl._fetch_one(self.disk_idx, int(src.item()))
            self.ctrl.copied_bytes += EXPERT_BYTES
            comp = {name: [mx.concatenate([self.hot[name][j], staged[name][j]], axis=0)
                           for j in range(3)] for name in PROJ}
            new_hot = {name: [mx.take(comp[name][j], idx_update, axis=0)
                              for j in range(3)] for name in PROJ}
            self.hot = new_hot
            self.slot_mx = new_slot
            y = self._glu(x, new_hot, row_map.reshape(1, 1, self.k))
            return (y * scores[..., None]).sum(axis=-2)

        if mode == "floor":
            N, R, E = self.ctrl.N, self.R, self.E
            mx.eval(sel)                                           # fetch-on-miss causality
            if N > 0:
                src_ids = [(self._cycle + i * 7919) % E for i in range(N)]
                self._cycle = (self._cycle + N) % E
                # N>=2: one pread per expert in parallel; N==1: split the single
                # 663 KB read into QD/2 sub-reads (same steelman deploy uses, so
                # floor_n=1 and deploy move 1 expert/layer at the same read cost).
                staged = (self.ctrl._fetch_one(self.disk_idx, src_ids[0]) if N == 1
                          else self.ctrl._fetch(self.disk_idx, src_ids))
                self.ctrl.copied_bytes += N * EXPERT_BYTES
                comp = {name: [mx.concatenate([self.hot[name][j], staged[name][j]], axis=0)
                               for j in range(3)] for name in PROJ}
                idx = mx.concatenate([mx.arange(R - N, dtype=mx.int32),
                                      R + mx.arange(N, dtype=mx.int32)])   # [R] reads all N staged
                y = self._glu(x, comp, idx.reshape(1, 1, R))
                self.hot = {name: [mx.take(comp[name][j], idx, axis=0)     # slot update (~12 MB)
                                   for j in range(3)] for name in PROJ}
            else:
                idx = mx.arange(R, dtype=mx.int32)
                y = self._glu(x, self.hot, idx.reshape(1, 1, R))
            return (y * scores[..., None]).sum(axis=-2)

        raise ValueError(f"unknown xl mode {mode!r}")


class XLController:
    """Attaches disk-expert XLLayers to a loaded disk_experts Model.

    mode: 'noswap' | 'floor' | 'deploy' | 'lazy_full'.
    R defaults to num_experts_per_tok (k). N is the floor swap count.
    exact_prefill: run full-pool MoE during prefill (gates only; the benchmark
    uses hot-only prefill since prefill values do not affect decode latency).
    """

    def __init__(self, model, config, model_dir, mode, R=None, N=0, exact_prefill=False):
        import fcntl
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        self.mode = mode
        self.N = N
        self.exact_prefill = exact_prefill
        self.copied_bytes = 0
        q = config["quantization"]
        self.gs, self.bits, self.qmode = q["group_size"], q["bits"], q.get("mode", "affine")
        k = model.args.num_experts_per_tok
        self.R = k if R is None else R

        self.disk_path = str(Path(model_dir) / "experts_flat.bin")
        self.disk_fd = os.open(self.disk_path, os.O_RDONLY)
        fcntl.fcntl(self.disk_fd, 48, 1)                 # F_NOCACHE
        self.disk_qd = int(os.environ.get("TEMPORAL_DISK_QD", "16"))
        self.disk_ex = ThreadPoolExecutor(max_workers=self.disk_qd)
        L = model.args.num_hidden_layers
        E = model.args.num_experts
        fsz = os.fstat(self.disk_fd).st_size
        assert fsz == L * E * EXPERT_BYTES, (fsz, L * E * EXPERT_BYTES)
        self.disk_E = E
        self.layer_stride = E * EXPERT_BYTES             # bytes per layer on disk

        self.layers = []
        for i, layer in enumerate(model.model.layers):
            xl = XLLayer(layer.mlp, self, self.R, i)
            layer.mlp.xl = xl
            self.layers.append(xl)

    # ---- disk fetch (real preads -> parsed q4 tensors) ----
    def _fetch(self, layer_idx, expert_ids):
        eb, base = EXPERT_BYTES, layer_idx * self.layer_stride
        offs = [base + e * eb for e in expert_ids]
        def rd(o):
            b = os.pread(self.disk_fd, eb, o)
            assert len(b) == eb, f"short pread at offset {o}: {len(b)} != {eb}"
            return b
        if len(offs) == 1:
            data = rd(offs[0])
        else:
            data = b"".join(self.disk_ex.map(rd, offs))
        return parse_experts(data, len(expert_ids))

    def _fetch_one(self, layer_idx, expert_id):
        """Single-expert fetch (deploy) split into QD/2 parallel sub-reads --
        the steelman a competent engine would use for one 663 KB miss."""
        eb = EXPERT_BYTES
        off = layer_idx * self.layer_stride + expert_id * eb
        sub = max(1, self.disk_qd // 2)
        step = -(-eb // sub)
        parts = [(off + j * step, min(step, eb - j * step)) for j in range(sub)]
        def rd(p):
            b = os.pread(self.disk_fd, p[1], p[0])
            assert len(b) == p[1], f"short pread at offset {p[0]}: {len(b)} != {p[1]}"
            return b
        data = b"".join(self.disk_ex.map(rd, parts))
        return parse_experts(data, 1)

    def reset(self):
        self.copied_bytes = 0
        for xl in self.layers:
            xl.reset()

    def disable(self):
        for layer in self.layers:
            layer.block.xl = None
        try:
            self.disk_ex.shutdown(wait=False)
            os.close(self.disk_fd)
        except OSError:
            pass
