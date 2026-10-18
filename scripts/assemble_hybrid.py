#!/usr/bin/env python3
"""Assemble the DeepSeek-V4.1-Flash-1017 hybrid checkpoint in place.

Usage: assemble_hybrid.py BASE_DIR EXL3_DIR
  BASE_DIR  deepseek-ai/DeepSeek-V4.1-Flash-1017 @1bae0e10: config.json, model.safetensors.index.json,
            tokenizer files, Engram shards 47/48 and the nonexpert-*.safetensors from fetch_nonexpert.py
  EXL3_DIR  LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw @e55cd9e0 (full download)

Copies the routed-expert tensors of EXL3_DIR into BASE_DIR/exl3experts-*.safetensors, verifies the
Engram shards, and rewrites BASE_DIR's index and config (originals kept as *.base). BASE_DIR is then
the model directory to serve; EXL3_DIR can be deleted afterwards.
"""
import hashlib, json, os, re, shutil, struct, sys

ENGRAM_SHA = {
    "model-00047-of-00048.safetensors": "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5",
    "model-00048-of-00048.safetensors": "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd",
}
ROUTED = re.compile(r"layers\.(\d+)\.ffn\.experts\.\d+\.w[123]\.(trellis|suh|svh|mul1)$")
BASE_ROUTED = re.compile(r"layers\.\d+\.ffn\.experts\.")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(64 << 20):
            d.update(b)
    return d.hexdigest()


def extract(src, dst, keep):
    h, base = header(src)
    names = sorted((k for k in h if keep(k)), key=lambda k: h[k]["data_offsets"][0])
    if not names:
        return {}
    new, off = {}, 0
    for k in names:
        s, e = h[k]["data_offsets"]
        new[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + e - s]}
        off += e - s
    hb = json.dumps(new, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
        for k in names:
            s, e = h[k]["data_offsets"]
            fi.seek(base + s)
            left = e - s
            while left:
                chunk = fi.read(min(left, 256 << 20))
                fo.write(chunk); left -= len(chunk)
    os.replace(dst + ".part", dst)
    return {k: os.path.basename(dst) for k in names}


def main():
    out, exl3 = sys.argv[1], sys.argv[2]
    for name, want in ENGRAM_SHA.items():
        got = sha256(os.path.join(out, name))
        print(name, "OK" if got == want else f"MISMATCH {got}", flush=True)
        if got != want:
            sys.exit(1)
    idx_path = os.path.join(out, "model.safetensors.index.json")
    if not os.path.exists(idx_path + ".base"):
        shutil.copy(idx_path, idx_path + ".base")
    base_index = json.load(open(idx_path + ".base"))["weight_map"]
    weight_map = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("nonexpert-") and f.endswith(".safetensors"):
            for k in header(os.path.join(out, f))[0]:
                weight_map[k] = f
    for k, f in base_index.items():
        if f in ENGRAM_SHA:
            weight_map[k] = f
    missing = [k for k in base_index if not BASE_ROUTED.match(k) and k not in weight_map]
    if missing:
        sys.exit(f"{len(missing)} base tensors missing from the range fetch, e.g. {missing[:3]}")
    exl3_index = json.load(open(os.path.join(exl3, "model.safetensors.index.json")))["weight_map"]
    layers = sorted({int(m.group(1)) for k in exl3_index if (m := ROUTED.match(k))})
    for f in sorted(set(exl3_index.values())):
        dst = os.path.join(out, f.replace("model-", "exl3experts-"))
        got = ({k: os.path.basename(dst) for k in header(dst)[0]} if os.path.exists(dst)
               else extract(os.path.join(exl3, f), dst, lambda k: ROUTED.match(k) is not None))
        weight_map.update(got)
        print(f, len(got), "expert tensors", flush=True)
    n_routed = sum(1 for k in weight_map if ROUTED.match(k))
    if n_routed != len(layers) * 384 * 3 * 4:
        sys.exit(f"routed EXL3 tensors {n_routed} != {len(layers) * 384 * 3 * 4}")
    cfg_path = os.path.join(out, "config.json")
    if not os.path.exists(cfg_path + ".base"):
        shutil.copy(cfg_path, cfg_path + ".base")
    cfg = json.load(open(cfg_path + ".base"))
    q = json.load(open(os.path.join(exl3, "config.json")))["quantization_config"]
    cfg["quantization_config"]["exl3_routed_experts"] = {
        "layers": layers,
        "source": "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw@e55cd9e0b923e8a0c7895afb379d4d28fa503bbd",
        "codebook": q.get("codebook"), "version": q.get("version"), "out_scales": q.get("out_scales"),
    }
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    json.dump({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, open(idx_path, "w"), indent=1)
    os.makedirs(os.path.join(out, "exl3_source"), exist_ok=True)
    for f in ("README.md", "config.json", "exl3_k_map.json", "quantization_config.json"):
        shutil.copy(os.path.join(exl3, f), os.path.join(out, "exl3_source", f))
    print(f"DONE {len(weight_map)} tensors, EXL3 experts on layers {layers[0]}..{layers[-1]}", flush=True)


main()
