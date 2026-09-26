# GEMV FP16 activation reuse (xn from rmsnorm) — REJECT

Base: main `b9defbe` (post MoE-down FP16 staging). Short-context experiment 2
from `docs/research/decode-optimization-2026-09-26.md`.

## Change under test

- `add_rmsnorm` optionally writes `_Float16* out_h` (same rounding as GEMV `load_slot`).
- Decode scratch `xn_h[2048]`; K=2048 GEMVs that read mixer `xn` (qkv / z_ab / q,k,v / lm_head)
  consume `x_h` instead of re-converting FP32 in every wave.
- K=4096 out projections (`ssm_out`, attn `o`) left on the float path (not covered).
- Weight layout, R, grid unchanged. Bit-exact vs float path (`test_gemv_fp16_act`).

## Correctness

`test_gemv_fp16_act`, `test_q4k_gemv`, `test_q6k_gemv`, `test_model`, `test_moe`,
`test_fused_gemv`: all pass. First generated ids unchanged in e2e logs.

## Timing (ADR_002 D12)

Unpaired candidate medians looked like a small gain (~144.1 → 145.4, +0.9%) but
ranges overlapped the baseline. Paired alternating binaries (`pair-*-{baseline,candidate}.txt`):

| pair | baseline | candidate |
|---:|---:|---:|
| 1 | 146.9 | 145.6 |
| 2 | 144.1 | 145.6 |
| 3 | 146.0 | 147.1 |
| **median** | **146.0** | **145.6** |

**gain −0.27%** vs ≥2% gate → **REJECT**.

pp512 unchanged (~1440). Production wiring reverted; kernel `x_h` path and equivalence
tests kept for a follow-up that covers out/core producers (K=4096) or a stronger pack.

## Why this differs from MoE-down KEEP

Down stages once into LDS and every wave in the same workgroup reuses it, cutting LDS
traffic and conversion inside a hot kernel. Dense GEMV conversion was already once per
wave across rows; xn reuse only removes cross-wave duplicate global converts. Covering
only K=2048 from `xn` leaves ~12% of decode in K=4096 out GEMVs untouched.
