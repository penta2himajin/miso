#!/usr/bin/env python3
"""Summarise tensor types from a GGUF header, which may be a truncated prefix of the file.

Lets remote GGUFs be inspected without downloading the tensor data:
    curl -sL -r 0-67108863 -o /tmp/head.gguf <hf resolve url>
    tools/gguf_header.py /tmp/head.gguf
"""
import collections
import re
import struct
import sys

# ggml_type id -> (name, block elements, block bytes)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17),
}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("Q")
        s = self.b[self.p:self.p + n]
        self.p += n
        return s.decode("utf-8", "replace")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("I"), self.take("Q")
            if et in SCALAR:
                self.p += struct.calcsize("<" + SCALAR[et]) * n
            else:
                for _ in range(n):
                    self.value(et)
            return f"<array {n}>"
        return self.take(SCALAR[t])


def main(path):
    r = Reader(open(path, "rb").read())
    assert r.take("I") == 0x46554747, "not a GGUF file"
    version, n_tensors, n_kv = r.take("I"), r.take("Q"), r.take("Q")
    meta = {}
    for _ in range(n_kv):
        k = r.string()
        meta[k] = r.value(r.take("I"))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.take("Q") for _ in range(r.take("I"))]
        ty = r.take("I")
        r.take("Q")  # data offset
        tensors.append((name, dims, ty))

    print(f"GGUF v{version}: {n_tensors} tensors, file_type={meta.get('general.file_type')}, "
          f"name={meta.get('general.name')}")
    agg = collections.defaultdict(lambda: [0, 0])
    for name, dims, ty in tensors:
        tname, be, bb = TYPES.get(ty, (f"type{ty}", 1, 0))
        n = 1
        for d in dims:
            n *= d
        kind = re.sub(r"blk\.\d+\.", "blk.N.", name)
        a = agg[(kind, tname)]
        a[0] += 1
        a[1] += n // be * bb
    by_type = collections.Counter()
    for (kind, tname), (c, b) in sorted(agg.items()):
        by_type[tname] += b
        print(f"{kind:40s} {tname:7s} x{c:3d} {b / 2**20:9.1f} MiB")
    print("bytes by type:", ", ".join(f"{t} {b / 2**30:.2f} GiB" for t, b in by_type.most_common()))


if __name__ == "__main__":
    main(sys.argv[1])
