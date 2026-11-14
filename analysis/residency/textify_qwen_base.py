"""Write the raw multimodal Qwen3.5 checkpoint as a text-only per-expert checkpoint (the layout the
merged adapters use), with NO adapter: model.language_model.X -> model.X, fused experts split per
expert, vision tower dropped, one shard per layer. Purpose: evaluate the same base weights under
vLLM's Qwen3_5MoeForCausalLM (text class) to measure the class confound against the raw dir
(Qwen3_5MoeForConditionalGeneration), which every historical qwen base arm used."""
import json, struct, glob, re, sys, os, shutil, collections, torch
from safetensors.torch import save_file
src, dst, cfg_from = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
idx = {}
for f in sorted(glob.glob(f"{src}/*.safetensors")):
    with open(f, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(hlen)); hdr.pop("__metadata__", None)
        for k, v in hdr.items(): idx[k] = (f, 8 + hlen, v)
def read(k):
    f, base, v = idx[k]; o0, o1 = v["data_offsets"]
    with open(f, "rb") as fh: fh.seek(base + o0); buf = fh.read(o1 - o0)
    return torch.frombuffer(bytearray(buf), dtype=DT[v["dtype"]]).reshape(v["shape"]).clone()
groups = collections.defaultdict(list)
for k in idx:
    if k.startswith("model.visual") or "mtp" in k: continue
    m = re.search(r"layers\.(\d+)\.", k); groups[int(m.group(1)) if m else -1].append(k)
weight_map = {}
for g in sorted(groups):
    out = {}
    for k in groups[g]:
        nk = re.sub(r"^model\.language_model\.", "model.", k); t = read(k)
        if nk.endswith("mlp.experts.gate_up_proj"):            # (E,2I,H) -> per expert gate/up (I,H)
            E, I2, H = t.shape; I = I2 // 2; pre = nk[: -len("gate_up_proj")]
            for e in range(E): out[f"{pre}{e}.gate_proj.weight"] = t[e, :I].contiguous(); out[f"{pre}{e}.up_proj.weight"] = t[e, I:].contiguous()
        elif nk.endswith("mlp.experts.down_proj"):             # (E,H,I) -> per expert (H,I)
            pre = nk[: -len("down_proj")]
            for e in range(t.shape[0]): out[f"{pre}{e}.down_proj.weight"] = t[e].contiguous()
        else: out[nk] = t
    name = f"model-layer{g:03d}.safetensors" if g >= 0 else "model-shared.safetensors"
    save_file(out, f"{dst}/{name}", metadata={"format": "pt"})
    for k in out: weight_map[k] = name
    print(f"[textify] {name}: {len(out)} tensors", flush=True)
json.dump({"metadata": {}, "weight_map": weight_map}, open(f"{dst}/model.safetensors.index.json", "w"), indent=1)
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
    if os.path.exists(f"{cfg_from}/{f}"): shutil.copy(f"{cfg_from}/{f}", f"{dst}/{f}")
print(f"[textify] done: {len(weight_map)} tensors -> {dst}", flush=True)
