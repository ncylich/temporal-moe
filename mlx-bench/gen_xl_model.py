#!/usr/bin/env python3
"""Deterministic (seed 0) builder for the bigger-than-RAM "xl" MoE model.

Same fine geometry as qwen3moe-rand-fine-q4 (hidden 1024, 45 layers, 8/4 heads
head_dim 128, vocab 151669, top-18, moe_ff 384, q4 g64) but E=1024 experts per
layer. The expert pool (45 x 1024 x 663552 B = 30.6 GB of REAL quantized random
weights) lives ON DISK in `experts_flat.bin` and is fetched into the compute path
at decode time (xl.py); all-resident is physically impossible. Non-expert weights
(attention, norms, embeddings, router gates) stay in RAM in `nonexpert.safetensors`.

On-disk layout of experts_flat.bin: [layer][expert], each expert = 663552 bytes =
concat over PROJ=(gate_proj, up_proj, down_proj) of  weight | scales | biases:

  gate_proj / up_proj : [384,1024] q4 g64
      weight  uint32  [384,128]  196608 B
      scales  f16     [384, 16]   12288 B
      biases  f16     [384, 16]   12288 B
  down_proj           : [1024,384] q4 g64
      weight  uint32  [1024, 48] 196608 B
      scales  f16     [1024,  6]  12288 B
      biases  f16     [1024,  6]  12288 B

Byte sub-offsets within one expert are the module-level SUB_OFFSETS dict below;
xl.py's fetch path and tests/g3_xl_exactness.py import it.

Determinism: expert weights are seeded PER LAYER (EXPERT_SEED + li) and drawn in
fixed chunk order, so a truncated L=3 build and the full L=45 build share layers
0..2 bytewise, and a disk build and an in-RAM build from the same code are
bytewise-equal. Non-expert weights use the gen_random_qwen3moe recipe (seed 0,
normal(0, STD), q4 g64, q8 router gate), drawn once at the top.

Usage:
  python gen_xl_model.py            # full L=45 E=1024 disk model -> models/qwen3moe-rand-xl-q4-disk/
  python gen_xl_model.py --dry      # print sizes/offsets only, generate nothing
"""
import argparse
import gc
import json
import os
import resource
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent

# ---- geometry (fine) ----
H = 1024
FF = 384
NH, NKV, HD = 8, 4, 128
VOCAB = 151669
STD = 0.02
GROUP, BITS, GATE_BITS = 64, 4, 8
K_TOP = 18

NONEXPERT_SEED = 0
EXPERT_SEED = 0            # per-layer key = EXPERT_SEED + li
EXPERT_CHUNK = 128         # experts generated per chunk (bounds RAM; locked for determinism)

PROJ = ("gate_proj", "up_proj", "down_proj")

# per-projection quant-tensor (shape, numpy dtype) for the on-disk bytes
_TSHAPE = {
    "gate_proj": {"weight": ((FF, H // 8), np.uint32),
                  "scales": ((FF, H // GROUP), np.float16),
                  "biases": ((FF, H // GROUP), np.float16)},
    "up_proj":   {"weight": ((FF, H // 8), np.uint32),
                  "scales": ((FF, H // GROUP), np.float16),
                  "biases": ((FF, H // GROUP), np.float16)},
    "down_proj": {"weight": ((H, FF // 8), np.uint32),
                  "scales": ((H, FF // GROUP), np.float16),
                  "biases": ((H, FF // GROUP), np.float16)},
}

# SUB_OFFSETS[name][tensor] = (byte_offset, nbytes, shape, np_dtype)
SUB_OFFSETS = {}
_o = 0
for _name in PROJ:
    SUB_OFFSETS[_name] = {}
    for _t in ("weight", "scales", "biases"):
        _shape, _dt = _TSHAPE[_name][_t]
        _nb = int(np.prod(_shape)) * np.dtype(_dt).itemsize
        SUB_OFFSETS[_name][_t] = (_o, _nb, _shape, _dt)
        _o += _nb
EXPERT_BYTES = _o
assert EXPERT_BYTES == 663552, EXPERT_BYTES


def parse_experts(data, n):
    """Parse `n` contiguous experts' raw bytes -> {name: (wq, sc, bi)} mx arrays.

    Inverse of chunk_to_bytes: reinterprets each sub-slice per SUB_OFFSETS with
    the native (little-endian) dtype, so on the same machine the round-trip is
    an identity (uint32 weights, float16 scales/biases)."""
    a = np.frombuffer(data, dtype=np.uint8).reshape(n, EXPERT_BYTES)
    out = {}
    for name in PROJ:
        parts = []
        for t in ("weight", "scales", "biases"):
            off, nb, shape, dt = SUB_OFFSETS[name][t]
            col = np.ascontiguousarray(a[:, off:off + nb]).view(dt).reshape(n, *shape)
            parts.append(mx.array(col))
        out[name] = tuple(parts)
    return out


def chunk_to_bytes(chunk_experts):
    """{name: (wq, sc, bi)} for a chunk of experts -> flat bytes (expert-major)."""
    n = chunk_experts["gate_proj"][0].shape[0]
    cols = []
    for name in PROJ:
        for tensor in chunk_experts[name]:          # wq, sc, bi in order
            cols.append(np.array(mx.view(tensor.reshape(n, -1), dtype=mx.uint8)))
    flat = np.concatenate(cols, axis=1)
    assert flat.shape == (n, EXPERT_BYTES), flat.shape
    return flat.tobytes()


def gen_layer_experts(li, E, chunk=EXPERT_CHUNK):
    """Yield (e0, n, {name: (wq, sc, bi)}) chunks for one layer. Deterministic:
    seeded per layer, fixed chunk order/size -> chunking never changes bytes."""
    mx.random.seed(EXPERT_SEED + li)
    dims = {"gate_proj": (FF, H), "up_proj": (FF, H), "down_proj": (H, FF)}
    for e0 in range(0, E, chunk):
        n = min(chunk, E - e0)
        out = {}
        for name in PROJ:
            out_dim, in_dim = dims[name]
            w = (mx.random.normal((n, out_dim, in_dim)) * STD).astype(mx.float16)
            wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=BITS, mode="affine")
            mx.eval(wq, sc, bi)
            out[name] = (wq, sc, bi)
            del w
        yield e0, n, out


def _base_config(L, E, disk):
    cfg = dict(
        model_type="qwen3_moe",
        hidden_size=H,
        num_hidden_layers=L,
        num_attention_heads=NH,
        num_key_value_heads=NKV,
        head_dim=HD,
        vocab_size=VOCAB,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        tie_word_embeddings=True,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        num_experts=E,
        num_experts_per_tok=K_TOP,
        moe_intermediate_size=FF,
        intermediate_size=FF,
    )
    if disk:
        cfg["disk_experts"] = True
    return cfg


def gen_nonexpert(L, E):
    """Non-expert weights (embeddings, attention, norms, router gates) + the
    per-layer q8 gate override map. Recipe/stds mirror gen_random_qwen3moe."""
    mx.random.seed(NONEXPERT_SEED)
    weights = {}

    def quant(key, shape, bits):
        w = (mx.random.normal(shape) * STD).astype(mx.float16)
        wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=bits, mode="affine")
        mx.eval(wq, sc, bi)
        weights[f"{key}.weight"] = wq
        weights[f"{key}.scales"] = sc
        weights[f"{key}.biases"] = bi

    def norm(key, dim):
        weights[f"{key}.weight"] = mx.ones((dim,), dtype=mx.float16)

    quant("model.embed_tokens", (VOCAB, H), BITS)
    gate_overrides = {}
    for li in range(L):
        p = f"model.layers.{li}"
        quant(f"{p}.self_attn.q_proj", (NH * HD, H), BITS)
        quant(f"{p}.self_attn.k_proj", (NKV * HD, H), BITS)
        quant(f"{p}.self_attn.v_proj", (NKV * HD, H), BITS)
        quant(f"{p}.self_attn.o_proj", (H, NH * HD), BITS)
        norm(f"{p}.self_attn.q_norm", HD)
        norm(f"{p}.self_attn.k_norm", HD)
        norm(f"{p}.input_layernorm", H)
        norm(f"{p}.post_attention_layernorm", H)
        quant(f"{p}.mlp.gate", (E, H), GATE_BITS)
        gate_overrides[f"{p}.mlp.gate"] = {"group_size": GROUP, "bits": GATE_BITS}
    norm("model.norm", H)
    return weights, gate_overrides


def _write_config(out_dir, cfg, gate_overrides, extra):
    cfg = dict(cfg)
    cfg["quantization"] = {"group_size": GROUP, "bits": BITS, **gate_overrides}
    cfg["_build"] = {"seed": 0, "std": STD, "per_expert_bytes": EXPERT_BYTES, **extra}
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def write_disk_model(out_dir, L=45, E=1024, chunk=EXPERT_CHUNK):
    """Full disk-experts model: nonexpert.safetensors + experts_flat.bin + config."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _base_config(L, E, disk=True)
    t0 = time.perf_counter()

    weights, gate_overrides = gen_nonexpert(L, E)
    mx.save_safetensors(str(out_dir / "nonexpert.safetensors"), weights)
    nonexpert_bytes = sum(a.nbytes for a in weights.values())
    del weights
    gc.collect()

    flat_path = out_dir / "experts_flat.bin"
    written = 0
    with open(flat_path, "wb") as f:
        for li in range(L):
            for e0, n, ch in gen_layer_experts(li, E, chunk):
                b = chunk_to_bytes(ch)
                f.write(b)
                written += len(b)
            if li % 5 == 0 or li == L - 1:
                gc.collect()
                print(f"  [xl] layer {li + 1}/{L}  written={written / 1e9:.2f} GB  "
                      f"RSS={rss_gib():.2f} GiB", flush=True)

    expected = L * E * EXPERT_BYTES
    assert written == expected, (written, expected)
    assert flat_path.stat().st_size == expected, flat_path.stat().st_size

    _write_config(out_dir, cfg, gate_overrides,
                  {"num_experts": E, "disk_expert_pool_bytes": expected,
                   "nonexpert_bytes": nonexpert_bytes})
    dt = time.perf_counter() - t0
    print(f"[xl] L={L} E={E}  experts_flat.bin={expected / 1e9:.3f} GB  "
          f"nonexpert={nonexpert_bytes / 1e9:.3f} GB  build={dt:.1f}s  "
          f"peakRSS={rss_gib():.2f} GiB  -> {out_dir}", flush=True)
    return dict(disk_bytes=expected, build_s=dt, peak_rss_gib=rss_gib())


def write_inram_model(out_dir, L=3, E=1024, chunk=EXPERT_CHUNK):
    """Ordinary in-RAM model (full E-expert switch_mlp stacks) from the SAME
    generators -> loadable by model.load. Used only by the g3 exactness gate;
    its expert bytes are identical to write_disk_model's for the same L, E."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _base_config(L, E, disk=False)

    weights, gate_overrides = gen_nonexpert(L, E)
    for li in range(L):
        acc = {name: ([], [], []) for name in PROJ}
        for e0, n, ch in gen_layer_experts(li, E, chunk):
            for name in PROJ:
                for j in range(3):
                    acc[name][j].append(ch[name][j])
        for name in PROJ:
            wq = mx.concatenate(acc[name][0], axis=0)
            sc = mx.concatenate(acc[name][1], axis=0)
            bi = mx.concatenate(acc[name][2], axis=0)
            mx.eval(wq, sc, bi)
            key = f"model.layers.{li}.mlp.switch_mlp.{name}"
            weights[f"{key}.weight"] = wq
            weights[f"{key}.scales"] = sc
            weights[f"{key}.biases"] = bi

    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
    total = sum(a.nbytes for a in weights.values())
    _write_config(out_dir, cfg, gate_overrides, {"num_experts": E, "total_quant_bytes": total})
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "qwen3moe-rand-xl-q4-disk"))
    ap.add_argument("--layers", type=int, default=45)
    ap.add_argument("--experts", type=int, default=1024)
    ap.add_argument("--dry", action="store_true",
                    help="print byte layout + total size and exit")
    args = ap.parse_args()

    total = args.layers * args.experts * EXPERT_BYTES
    print(f"per_expert_bytes = {EXPERT_BYTES}")
    print("SUB_OFFSETS:")
    for name in PROJ:
        for t in ("weight", "scales", "biases"):
            off, nb, shape, dt = SUB_OFFSETS[name][t]
            print(f"  {name:10s}.{t:6s} off={off:6d} nb={nb:6d} shape={shape} {np.dtype(dt).name}")
    print(f"experts_flat.bin = {total} B = {total / 1e9:.3f} GB "
          f"(L={args.layers} x E={args.experts})")
    if args.dry:
        return

    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    print(f"free disk = {free / 1e9:.1f} GB")
    if free < total + 1e9:
        raise SystemExit(f"insufficient disk: need ~{(total + 1e9) / 1e9:.0f} GB, "
                         f"have {free / 1e9:.1f} GB (delete the scratchpad disk_pool)")
    write_disk_model(args.out, L=args.layers, E=args.experts)


if __name__ == "__main__":
    main()
