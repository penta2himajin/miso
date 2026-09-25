#!/usr/bin/env python3
"""Generate test fixtures for the GGUF reader and CPU dequantisation (milestone M1).

Writes into tests/fixtures/:
  tiny.gguf              small GGUF with one metadata key per value type and F32/F16/BF16/Q4_K/Q6_K
                         tensors; the Q4_K/Q6_K blocks are copied from the real model
  tiny.<tensor>.f32      gguf-py dequantisation of each tensor (the bit-exact reference)
  ornith-q4km-tensors.tsv  name, type, dims (GGUF order), absolute offset, bytes for every tensor
                         of the model GGUF

Usage: PYTHONPATH=<llama.cpp>/gguf-py tools/make_test_fixtures.py <Ornith-1.5-35B-Q4_K_M.gguf>
"""
import os
import sys

import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf import GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import dequantize

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
BLOCKS = 64  # blocks of real Q4_K / Q6_K data per fixture tensor


def raw_blocks(reader, name, n_blocks, block_bytes):
    t = next(t for t in reader.tensors if t.name == name)
    raw = np.asarray(t.data).view(np.uint8).reshape(-1)
    return raw[: n_blocks * block_bytes].copy()


def write_manifest(reader, path):
    with open(path, "w") as f:
        f.write("# name\ttype\tdims(GGUF order)\tabs_offset\tbytes\n")
        for t in reader.tensors:
            dims = ",".join(str(int(d)) for d in t.shape)
            f.write(f"{t.name}\t{int(t.tensor_type)}\t{dims}\t{int(t.data_offset)}\t{int(t.n_bytes)}\n")


def main(model_path):
    os.makedirs(OUT, exist_ok=True)
    model = GGUFReader(model_path)
    write_manifest(model, os.path.join(OUT, "ornith-q4km-tensors.tsv"))

    rng = np.random.default_rng(20260925)
    tensors = {
        # name: (numpy array as stored, raw_dtype or None)
        "f32": (rng.standard_normal((5, 3)).astype(np.float32), None),
        "f16": (rng.standard_normal(256).astype(np.float16), None),
        "bf16": ((rng.standard_normal(64).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16),
                 QT.BF16),
        "q4_k": (raw_blocks(model, "blk.0.ffn_gate_exps.weight", BLOCKS, 144).reshape(2, -1), QT.Q4_K),
        "q6_k": (raw_blocks(model, "output.weight", BLOCKS, 210).reshape(4, -1), QT.Q6_K),
    }

    w = GGUFWriter(os.path.join(OUT, "tiny.gguf"), "miso-test")
    w.add_custom_alignment(64)
    w.add_uint8("test.u8", 200)
    w.add_int8("test.i8", -100)
    w.add_uint16("test.u16", 60000)
    w.add_int16("test.i16", -30000)
    w.add_uint32("test.u32", 4000000000)
    w.add_int32("test.i32", -2000000000)
    w.add_uint64("test.u64", 1 << 40)
    w.add_int64("test.i64", -(1 << 40))
    w.add_float32("test.f32", 0.25)
    w.add_float64("test.f64", -1.5)
    w.add_bool("test.bool", True)
    w.add_string("test.string", "miso ミソ")
    w.add_array("test.array_i32", [1, -2, 3])
    w.add_array("test.array_str", ["a", "bc", ""])
    for name, (arr, raw_dtype) in tensors.items():
        w.add_tensor(name, arr, raw_dtype=raw_dtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    tiny = GGUFReader(os.path.join(OUT, "tiny.gguf"))
    for t in tiny.tensors:
        ref = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(-1)
        ref.tofile(os.path.join(OUT, f"tiny.{t.name}.f32"))
        print(f"{t.name}: type={t.tensor_type.name} dims={list(map(int, t.shape))} "
              f"offset={int(t.data_offset)} elements={ref.size}")
    assert tiny.fields["test.array_i32"].types == [GGUFValueType.ARRAY, GGUFValueType.INT32]


if __name__ == "__main__":
    main(sys.argv[1])
