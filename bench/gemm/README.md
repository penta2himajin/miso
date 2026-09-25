# Prefill GEMM benchmark

`bench_qgemm` measures the prefill GEMM of `kernels/qgemm.hpp` on real Ornith dense projections at T = 64 / 512 / 2048 tokens. It reports time per launch and TFLOP/s against the 25.4 TFLOP/s `v_dot2_f32_f16` peak, plus the FP16 activation prep as a separate launch.

```bash
./build/bench/bench_qgemm            # model path from MISO_ORNITH_GGUF
```

| File | Kernel version |
|---|---|
| `results/2026-09-26-run1.txt` | 64 x 64 tile, 4 x 4 per thread, K stage 64, next stage prefetched into registers, 2 workgroups per CU; `-smi.csv` clock log |

Findings behind this version (T = 2048, attn_q):

- v1 (no alignment on the LDS struct) compiled the operand reads to `ds_read2_b32`: 39.8% of peak. Aligning the struct to 16 B gives `ds_read_b128`: 45.7%.
- Prefetching the next stage into registers raised VGPRs to 132 (1 workgroup per CU) and lost time. With `__launch_bounds__(256, 2)`, VGPRs were capped at 128: 52%. Fully unrolled Q6_K then spilled 184 B to scratch (16%). Unrolling two chunks at a time removed all spills in both formats.
- Diagnostic variants: halving the weight LDS reads gains 8%, and dropping all loads after stage 0 reaches 62%. The main loop is therefore not LDS-bandwidth bound. The remainder is per-stage cost: barriers, the scale epilogue and the LDS stores.
- T = 64 reaches 20–31%: 64-token tiles leave most CUs idle for N <= 8192. It is still ~5x faster than 64 decode GEMVs (27.6 us per 13.8 MB Q6_K qkv launch).
