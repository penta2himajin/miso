# Decode GEMV benchmarks

`bench_q4k_gemv` (built by `cmake --build --preset default`) measures the Q4_K decode GEMV of `kernels/q4k_gemv.hpp` on real Ornith projections:

- accuracy of each activation format against an FP64 reference;
- per-launch time over layer-rotated weights, so every launch reads L2-cold data as in decode;
- `tall`: all layers of a projection in one launch (asymptotic bandwidth);
- `scale`: launch-size sweep at rotating offsets, to separate fixed cost from streaming.

```bash
./build/bench/bench_q4k_gemv                 # full run (model path from MISO_ORNITH_GGUF)
./build/bench/bench_q4k_gemv <gguf> --tall-only
```

| File | Kernel version |
|---|---|
| `results/2026-09-25-run1*.txt` | v1: GGUF scales, `ds_bpermute` reduction |
| `results/2026-09-25-run2-v2.txt` | v2: r1 scales, DPP reduction, FP16 offset folding |
| `results/2026-09-25-run3-rows.txt` | rows-per-wave with LDS/scratch spill (superseded; see ADR_004) |
| `results/2026-09-25-run4-rows.txt` | rows-per-wave, no spills; `run4-smi.csv` clock log |
| `results/2026-09-25-run5-scaling.txt` | launch-size scaling |

Decision and analysis: `docs/decisions/ADR_004-decode-gemv-activation-format.md`.
