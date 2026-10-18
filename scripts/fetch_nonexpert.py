#!/usr/bin/env python3
"""Range-fetch every tensor of DeepSeek-V4.1-Flash-1017 except the routed experts of the target
layers and the Engram tables, without downloading whole shards (~17 GiB instead of ~286 GiB).

Usage: fetch_nonexpert.py BASE_DIR
  BASE_DIR must already hold the base repo's model.safetensors.index.json (see the recipe).
  Writes nonexpert-000NN-of-00048.safetensors next to it; safe to re-run (skips finished files).
"""
import json, os, re, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO, REV = "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/" + "{}"
SKIP = re.compile(r"layers\.\d+\.(ffn\.experts\.|engram\.embed\.)")
ENGRAM_SHARDS = {"model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"}


def get(name, start, end):
    for _ in range(8):
        try:
            req = urllib.request.Request(URL.format(name), headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start:
                return data
        except Exception as e:  # retry transient HTTP errors
            print(f"retry {name} @{start}: {e}", flush=True)
    raise SystemExit(f"failed to fetch {name} @{start}")


def do_shard(dest, name):
    out = os.path.join(dest, name.replace("model-", "nonexpert-"))
    if os.path.exists(out):
        return name, 0
    n = struct.unpack("<Q", get(name, 0, 8))[0]
    header = json.loads(get(name, 8, 8 + n))
    meta = header.pop("__metadata__", None)
    keep = sorted((k for k in header if not SKIP.match(k)), key=lambda k: header[k]["data_offsets"][0])
    if not keep:
        return name, 0
    base, ranges, cur = 8 + n, [], None
    for k in keep:  # coalesce neighbours into <= 256 MiB range requests
        s, e = header[k]["data_offsets"]
        if cur and s - cur[1] <= 1 << 20 and e - cur[0] <= 256 << 20:
            cur[1] = e; cur[2].append(k)
        else:
            cur = [s, e, [k]]; ranges.append(cur)
    blobs = {}
    for s, e, ks in ranges:
        data = get(name, base + s, base + e)
        for k in ks:
            ks_, ke_ = header[k]["data_offsets"]
            blobs[k] = data[ks_ - s:ke_ - s]
    new, off = {}, 0
    for k in keep:
        new[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [off, off + len(blobs[k])]}
        off += len(blobs[k])
    if meta:
        new["__metadata__"] = meta
    hb = json.dumps(new, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for k in keep:
            f.write(blobs[k])
    os.replace(out + ".part", out)
    return name, off


def main():
    dest = sys.argv[1]
    idx = json.load(open(os.path.join(dest, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({v for k, v in idx.items() if not SKIP.match(k)} - ENGRAM_SHARDS)
    total = 0
    with ThreadPoolExecutor(6) as ex:
        for name, nbytes in ex.map(lambda s: do_shard(dest, s), shards):
            total += nbytes
            print(f"{name}: {nbytes / 2**20:.1f} MiB", flush=True)
    print(f"DONE {total / 2**30:.2f} GiB", flush=True)


main()
