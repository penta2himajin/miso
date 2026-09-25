# ADR_004: Decode GEMV activation format — FP16

- Status: Accepted
- Date: 2026-09-25
- Resolves: ADR_002 D9 (activation format A/B, decided at M2)
- Evidence: `bench/gemv/results/2026-09-25-run*.txt`, `tests/test_q4k_gemv.hip`

## Context

ADR_002 D9 left the decode GEMV activation format to measurement:

- **A**: activations quantised on the fly to per-32 absmax INT8, weights consumed via `v_dot4_i32_i8`.
- **B**: FP16 activations through `v_dot2_f32_f16`.

Both variants were built in `kernels/q4k_gemv.hpp` (one wavefront per row, Q4_K r1 weights) and compared on real Ornith projections with a real activation (`layer0.moe_in` from the golden file). Each launch reads weights that are not in L2; this is the decode access pattern.

## Measurements (MI50, ROCm 6.3.4, sclk 1725 MHz under load)

Accuracy against an FP64 reference, per row `|y − Wx| / Σ|W x|`:

| Projection | A (INT8) max / rms | B (FP16) max / rms |
|---|---|---|
| `attn_gate` 2048→4096 | 2.93e-3 / 1.15e-3 | 9.7e-5 / 3.7e-5 |
| `ssm_out` 4096→2048 | 1.96e-3 / 5.6e-4 | 9.3e-5 / 1.5e-5 |
| `attn_q` 2048→8192 | 3.25e-3 / 1.06e-3 | 8.8e-5 / 3.5e-5 |

Speed, after the v2 changes (r1 scales, DPP reduction, FP16 offset folding):

| Case | A (INT8) | B (FP16) |
|---|---|---|
| Tall 141.6 MB single launch, best grid | 691 GB/s (86.7%) | 666 GB/s (83.5%) |
| `attn_gate` 4.72 MB per launch, best of R ∈ {1,4}, grid ∈ {120,240} | 15.5 µs | 16.6 µs |
| `attn_q` 9.44 MB per launch, R = 4 | 21.0 µs | 23.4 µs |

Before v2, B was instruction-bound at 499 GB/s (63%). Its row loop was 150 instructions, against a ceiling of about 150 per 1,152-byte row at 797 GB/s: 60 CUs × 1 wave-instruction per cycle × 1.725 GHz.

## Decision

Decode GEMVs use **FP16 activations (variant B)**. Residual stream and accumulation stay FP32 (ADR_002 D9).

In these runs B is 4–12% slower per launch (7% on `attn_gate`, 11% on `attn_q`). Run-to-run variation on small launches is about ±1 µs: in run 2, B was faster than A on `attn_gate`. B's error is 21–37× lower. For the matrices decode reads once per token, launch fixed cost dominates the time (see below), not ALU throughput. That also makes the gap likely to shrink once projections are fused.

The INT8 variant stays in the code as a measured alternative. It is not used by the engine.

## Findings that shape the next steps

- **Per-launch fixed cost is ~7–9 µs** for a GEMV (`scale` rows in `run5-scaling.txt`): 0.55 MB takes 7 µs; the linear fit over 9–38 MB gives an 8.8 µs intercept and ~700 GB/s marginal bandwidth. A dependent read-one/write-one kernel costs ~3 µs back to back (`bench/mi50/results/2026-09-25-launch-grid.txt`). The rest is per-launch activation setup, first-access latency and drain.
- **Consequence for decode**: at ~12 GEMV-sized launches per layer this is ~3.4 ms/token of fixed cost alone, more than the 2.5 ms bandwidth floor. Fusing projections that share an input (DeltaNet `qkv`+`z`+`a`+`b`; expert gate+up; all 8 routed experts in one launch) is the first lever. This is the measured input to the ADR_001 D4 megakernel trigger.
- **Rows per wave (R)**: issuing R rows' loads before computing helps large launches by 5–10% (e.g. 600 → 659 GB/s at 38 MB) and does not help small ones. R is a per-shape tuning parameter.
- **Compiler pitfall**: selecting between struct members (`g < 2 ? h.y : …`) was turned into a dynamic offset into a stack copy, which cost LDS and scratch (visible in the kernel resource report). Selecting between computed values avoids it (`kernels/q4k_gemv.hpp`, `r1_scales`).
