#!/usr/bin/env python3
"""Report per-kernel register / LDS / scratch usage and occupancy from a built HIP binary.

Reads the AMDGPU code-object metadata (what the compiler actually allocated), so the numbers
match `-Rpass-analysis=kernel-resource-usage`.

Usage: tools/kernel_resources.py <executable-or-object> [--arch gfx906] [--out report.txt]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")
READELF = f"{ROCM}/lib/llvm/bin/llvm-readelf"
ROC_OBJ_LS = f"{ROCM}/bin/roc-obj-ls"

# gfx906 (GCN5): per SIMD 256 VGPRs (granule 4), 800 SGPRs (granule 16), 10 waves max;
# 4 SIMDs and 64 KiB LDS per CU.
MAX_WAVES_PER_SIMD, VGPRS, VGPR_GRANULE, SGPRS, SGPR_GRANULE = 10, 256, 4, 800, 16
SIMDS_PER_CU, LDS_PER_CU, WAVE = 4, 65536, 64


def ceil_to(x, g):
    return (x + g - 1) // g * g


def occupancy(k):
    """Waves per SIMD, accounting for whole workgroups being resident together."""
    per_simd = MAX_WAVES_PER_SIMD
    if k["vgpr_count"]:
        per_simd = min(per_simd, VGPRS // ceil_to(k["vgpr_count"], VGPR_GRANULE))
    if k["sgpr_count"]:
        per_simd = min(per_simd, SGPRS // ceil_to(k["sgpr_count"], SGPR_GRANULE))
    waves_per_wg = -(-k["max_flat_workgroup_size"] // WAVE)
    wgs = (per_simd * SIMDS_PER_CU) // waves_per_wg
    if k["group_segment_fixed_size"]:
        wgs = min(wgs, LDS_PER_CU // k["group_segment_fixed_size"])
    return min(per_simd, wgs * waves_per_wg // SIMDS_PER_CU)


def code_objects(path, arch):
    """Yield raw bytes of every code object for `arch` embedded in `path`."""
    sections = subprocess.run([READELF, "-S", "-W", path], capture_output=True, text=True,
                              check=True).stdout
    if ".hip_fatbin" not in sections:
        return  # host-only binary; roc-obj-ls fails on these
    out = subprocess.run([ROC_OBJ_LS, path], capture_output=True, text=True, check=True).stdout
    blob = open(path, "rb").read()
    for line in out.splitlines():
        m = re.search(r"--(gfx\w+)\S*\s+file://.*#offset=(\d+)&size=(\d+)", line)
        if m and m.group(1) == arch:
            off, size = int(m.group(2)), int(m.group(3))
            yield blob[off:off + size]


def kernels(co_bytes):
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(co_bytes)
        f.flush()
        notes = subprocess.run([READELF, "--notes", f.name], capture_output=True, text=True,
                               check=True).stdout
    fields = ("name", "vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
              "group_segment_fixed_size", "private_segment_fixed_size", "max_flat_workgroup_size")
    # Under `amdhsa.kernels:` each kernel starts with "  - .<key>:"; its own keys are indented by
    # exactly 4 spaces (deeper lines belong to .args and other nested lists).
    ks, cur = [], None
    for line in notes.splitlines():
        if line.startswith("amdhsa.") and not line.startswith("amdhsa.kernels"):
            cur = None
            continue
        m = re.match(r"(  - |    )\.(\w+):\s*(\S*)", line)
        if not m:
            continue
        lead, key, val = m.groups()
        if lead == "  - ":
            cur = {}
            ks.append(cur)
        if cur is not None and key in fields and val:
            cur[key] = val if key == "name" else int(val)
    return [k for k in ks if "name" in k]


def demangle(names):
    cxxfilt = shutil.which("c++filt")
    if not cxxfilt or not names:
        return names
    out = subprocess.run([cxxfilt], input="\n".join(names), capture_output=True, text=True).stdout
    return out.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--arch", default="gfx906")
    ap.add_argument("--out")
    a = ap.parse_args()

    ks = [k for co in code_objects(a.binary, a.arch) for k in kernels(co)]
    names = demangle([k["name"] for k in ks])
    lines = [f"# {os.path.basename(a.binary)} ({a.arch})",
             f"{'VGPR':>4} {'SGPR':>4} {'LDS':>6} {'scratch':>7} {'spill':>5} {'wg':>4} "
             f"{'occ':>3}  kernel"]
    for k, name in sorted(zip(ks, names), key=lambda x: x[1]):
        spill = k.get("vgpr_spill_count", 0) + k.get("sgpr_spill_count", 0)
        lines.append(f"{k['vgpr_count']:>4} {k['sgpr_count']:>4} "
                     f"{k['group_segment_fixed_size']:>6} {k['private_segment_fixed_size']:>7} "
                     f"{spill:>5} {k['max_flat_workgroup_size']:>4} {occupancy(k):>3}  {name}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
