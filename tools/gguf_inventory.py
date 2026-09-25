#!/usr/bin/env python3
"""Inventory an Ornith / qwen35moe GGUF: tensor types, per-token decode bytes, BF16-exact F32 tensors.

Usage: PYTHONPATH=<llama.cpp>/gguf-py tools/gguf_inventory.py <model.gguf>
"""
import collections
import re
import sys

import numpy as np
from gguf import GGUFReader

N_EXPERTS, TOP_K, VOCAB = 256, 8, 248320
MAIN_LAYERS = 40  # blk.40 is the MTP (nextn) layer
DELTANET_STATE_RW = 30 * 32 * 128 * 128 * 4 * 2  # 30 layers, 32 heads, 128x128 FP32, read + write
KV_BYTES_PER_TOKEN = 10 * 2 * 256 * 2 * 2  # 10 layers, 2 KV heads, dim 256, K+V, FP16
HBM_READ_GBPS = 797  # docs/research/mi50.md §5.2


def category(name):
    m = re.match(r"blk\.(\d+)\.", name)
    if m and int(m.group(1)) >= MAIN_LAYERS:
        return "mtp layer"
    if name == "token_embd.weight":
        return "embedding (1 row)"
    if "_exps" in name:
        return "routed experts (8/256)"
    if "shexp" in name and "gate_inp" not in name:
        return "shared expert"
    if "ffn_gate_inp" in name:
        return "router"
    if name.startswith("output"):
        return "lm head"
    return "attention + deltanet + norms"


def main(path):
    r = GGUFReader(path)
    types = collections.defaultdict(lambda: [0, 0])
    per_token = collections.Counter()
    bf16 = collections.defaultdict(lambda: [0, 0])
    for t in r.tensors:
        kind = re.sub(r"blk\.\d+\.", "blk.N.", t.name)
        a = types[(kind, t.tensor_type.name)]
        a[0] += 1
        a[1] += int(t.n_bytes)
        b = int(t.n_bytes)
        if "_exps" in t.name:
            b = b * TOP_K / N_EXPERTS
        if t.name == "token_embd.weight":
            b = b / VOCAB
        per_token[category(t.name)] += b
        if t.tensor_type.name == "F32":
            u = np.asarray(t.data).view(np.uint32)
            e = bf16[kind]
            e[0] += 1
            e[1] += bool(np.all((u & 0xFFFF) == 0))

    print("## tensors")
    for (kind, ty), (n, b) in sorted(types.items()):
        print(f"{kind:40s} {ty:5s} x{n:3d} {b / 2**20:9.1f} MiB")
    print(f"total {sum(b for _, b in types.values()) / 2**30:.2f} GiB")

    print("\n## decode bytes per token (main 40 layers)")
    main_b = sum(v for k, v in per_token.items() if k != "mtp layer")
    for k, v in per_token.most_common():
        share = "" if k == "mtp layer" else f"{100 * v / main_b:5.1f}%"
        print(f"{k:30s} {v / 1e6:8.1f} MB {share}")
    print(f"weights {main_b / 1e6:.1f} MB, deltanet state r/w {DELTANET_STATE_RW / 1e6:.1f} MB")
    for ctx in (0, 4096, 32768, 131072):
        tot = main_b + DELTANET_STATE_RW + ctx * KV_BYTES_PER_TOKEN
        print(f"ctx {ctx:6d}: {tot / 1e9:.3f} GB -> floor {HBM_READ_GBPS * 1e9 / tot:6.1f} tok/s")

    print("\n## F32 tensors exactly representable in BF16")
    for kind, (n, ok) in sorted(bf16.items()):
        print(f"{kind:40s} {ok}/{n}")


if __name__ == "__main__":
    main(sys.argv[1])
