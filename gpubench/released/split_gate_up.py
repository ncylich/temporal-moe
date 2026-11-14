#!/usr/bin/env python3
"""Rewrite a GGUF with every fused `blk.N.ffn_gate_up_exps.weight` [n_embd, 2*n_ff, E] split into
`ffn_gate_exps.weight` (first n_ff rows) and `ffn_up_exps.weight` (last n_ff rows), as build_moe_ffn's
merged path reads them. Metadata and every other tensor are copied unchanged. The streamed slot path needs
separate gate/up/down expert tensors; rows are independent, so a later quantization of the split file gives
the same quantized rows as the fused one. Run on an unquantized (F32/F16/BF16) file.

    PYTHONPATH=<llama.cpp>/gguf-py python3 split_gate_up.py in-bf16.gguf out-bf16.gguf
"""
import sys
import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails


def main():
    src, dst = sys.argv[1], sys.argv[2]
    r = gguf.GGUFReader(src, "r")
    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=st)
    out = []                                              # (name, numpy view, tensor type)
    nsplit = 0
    for t in r.tensors:
        if t.name.endswith("ffn_gate_up_exps.weight"):
            a = t.data                                    # numpy shape (E, 2*n_ff, row bytes or n_embd)
            h = a.shape[1] // 2
            assert a.shape[1] == 2 * h
            out.append((t.name.replace("ffn_gate_up_exps", "ffn_gate_exps"), a[:, :h], t.tensor_type))
            out.append((t.name.replace("ffn_gate_up_exps", "ffn_up_exps"), a[:, h:], t.tensor_type))
            nsplit += 1
        else:
            out.append((t.name, t.data, t.tensor_type))
    for name, a, tt in out:
        w.add_tensor_info(name, a.shape, a.dtype, a.nbytes, tt)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for name, a, tt in out:
        w.write_tensor_data(a.copy() if not a.flags["C_CONTIGUOUS"] else a, tensor_endianess=r.endianess)
    w.close()
    print(f"split {nsplit} fused gate_up tensors; wrote {len(out)} tensors to {dst}")


if __name__ == "__main__":
    main()
