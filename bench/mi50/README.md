# MI50 microbenchmarks

Probes and microbenchmarks behind `docs/research/mi50.md`.

| File | Purpose |
|---|---|
| `isa_probe.sh` | Compile one kernel per instruction builtin and report whether it lowers to native ISA |
| `microbench.hip` | HBM / cache / LDS bandwidth, ALU peaks, PCIe, launch overhead |
| `device_attrs.hip` | HIP device attributes relevant to engine design |
| `results/` | Raw output, one file per run (`YYYY-MM-DD-*.txt`) |

```bash
# Needed on this host: clang picks GCC 12, but only libstdc++-11-dev is installed.
export HIPFLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11"

bench/mi50/isa_probe.sh gfx906
hipcc $HIPFLAGS --offload-arch=gfx906 -O3 -o /tmp/mi50_microbench bench/mi50/microbench.hip && /tmp/mi50_microbench
hipcc $HIPFLAGS --offload-arch=gfx906 -O3 -o /tmp/mi50_attrs bench/mi50/device_attrs.hip && /tmp/mi50_attrs
```

Run on an idle GPU. Clocks follow the default `auto` power profile.
