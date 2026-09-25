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

## Amendment (2026-09-25, M3b): two accuracy defects fixed; decision unchanged

The first end-to-end DeltaNet test showed 2.0e-3 relative L2 error per token. That is 7× what an FP64 reference with FP16-rounded GEMV inputs predicts (2.96e-4). Isolating the `ssm_out` GEMV on the real activation (golden `mixer_core`, whose outputs cancel ~10×) found two causes:

1. **Inconsistent rounding in the Q4_K min term.** The kernel computed `d·sc·Σ q·x₁₆ − dmin·m·Σ x` with *exact* activation sums. Its FP16 rounding error therefore scaled with `d·sc·q` rather than `|W| = |d·sc·q − dmin·m|`, and exceeded the per-row bound where the two terms cancel. Now both terms use the converted activations (`Σ x₁₆`; INT8: `dx·Σ xq`), so the kernel computes exactly `Σ W·x₁₆`.
2. **Biased FP16 nibble encoding.** `0x6400|q = 1024 + q` made the FP32 chain accumulate a 1024× offset before subtracting it. Nibbles are now read as FP16 subnormals (`q·2⁻²⁴`, exact; gfx906 `v_dot2_f32_f16` keeps FP16 denormals) and the `2²⁴` scale is folded into `d`. This also removes the OR and the offset correction.

The GEMV tests gained a real-activation case (both output projections, 17 tokens). Their bound's chain term dropped from `1039·d·sc` to `15·d·sc` (Q4_K) and from `1087` to `95` (Q6_K).

| After the fix | INT8 | FP16 |
|---|---|---|
| rel. error rms (`attn_gate`, `moe_in` activation) | 4.74e-4 | 1.19e-5 |
| tall 141.6 MB launch, best | 687 GB/s | 675 GB/s |
| VGPRs (K = 2048, R = 1) | 47 | 45 |

The DeltaNet layer now matches the FP16-rounding prediction: 2.96e-4. FP16 remains the decode activation format; its accuracy margin over INT8 is now ~40×. Results: `bench/gemv/results/2026-09-25-run6-subnormal.txt`, `-q6k-run2-subnormal.txt`.

## Findings that shape the next steps

- **Per-launch fixed cost is ~7–9 µs** for a GEMV (`scale` rows in `run5-scaling.txt`): 0.55 MB takes 7 µs; the linear fit over 9–38 MB gives an 8.8 µs intercept and ~700 GB/s marginal bandwidth. A dependent read-one/write-one kernel costs ~3 µs back to back (`bench/mi50/results/2026-09-25-launch-grid.txt`). The rest is per-launch activation setup, first-access latency and drain.
- **Consequence for decode**: at ~12 GEMV-sized launches per layer this is ~3.4 ms/token of fixed cost alone, more than the 2.5 ms bandwidth floor. Fusing projections that share an input (DeltaNet `qkv`+`z`+`a`+`b`; expert gate+up; all 8 routed experts in one launch) is the first lever. This is the measured input to the ADR_001 D4 megakernel trigger.
- **Rows per wave (R)**: issuing R rows' loads before computing helps large launches by 5–10% (e.g. 600 → 659 GB/s at 38 MB) and does not help small ones. R is a per-shape tuning parameter.
- **Compiler pitfall**: selecting between struct members (`g < 2 ? h.y : …`) was turned into a dynamic offset into a stack copy, which cost LDS and scratch (visible in the kernel resource report). Selecting between computed values avoids it (`kernels/q4k_gemv.hpp`, `r1_scales`).
