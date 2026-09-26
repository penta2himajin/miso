# ADR_006: Decode execution stays kernel-per-operator (D4 megakernel not triggered)

- Status: Accepted
- Date: 2026-09-26
- Amends: ADR_001 D4 (megakernel trigger)
- Evidence: `bench/mi50/results/2026-09-26-m7e-grid-barrier.txt`, `bench/mi50/results/2026-09-26-m7e-decode-gaps.txt`

## Context

ADR_001 D4 adopted a persistent megakernel **only if** measurements taken while pursuing D4 showed it
faster, with the trigger being "measured inter-kernel overhead (dependent-kernel gaps × kernel count)
remaining a material share of token time after per-layer fusion". It also recorded that grid-barrier
cost on gfx906 was unmeasured and was itself a prerequisite. Two measurements were needed:

1. the cost of a grid-wide barrier (`cooperative_groups::this_grid().sync()`) relative to a kernel
   boundary on gfx906;
2. the share of decode time spent in dependent-kernel gaps after the M7a–M7d fusion work.

Per-layer fusion is now far along: the decode path issues a small number of fused kernels per layer,
with `qkv`/`z`/`a`/`b` and `q`/`k`/`v` combinations sharing a GEMV where their types allow (M7b,
M7c), so the launch count per token has already fallen by roughly a third.

## Measurements

**Grid barrier vs kernel boundary** (`bench/mi50/grid_barrier.hip`, idle GPU, 256-thread blocks):

| Grid | Grid barrier (`grid.sync()`) | Kernel boundary (separate launches) |
|---|---|---|
| 60 × 256 | 1.81 µs | 1.69 µs |
| 120 × 256 | 1.99 µs | 1.89 µs |
| 240 × 256 | 2.19 µs | 1.98 µs |
| 480 × 256 | 2.74 µs | 2.68 µs |

A barrier is within 0.1–0.2 µs of a boundary in the same shape, i.e. **not cheaper**. This is below
the 1.25–1.65 µs empty-launch cost of `microbench.hip` only because both probes measure a boundary
inside a dense back-to-back sequence, where dispatch is partly hidden.

A 20-step dependency chain over 256 floats (ping-pong buffers, each step reading the previous step's
global write) measured 3.21 µs/step as 20 stream launches versus 2.12 µs/step inside one cooperative
kernel at grid 60. The gap the megakernel closes is therefore ~1.1 µs/step — but only when each step
is a ~0.3–1.5 µs micro-operator and every step is a hard dependency.

**Dependent-kernel gaps in real decode** (`rocprof`, M7d build, full model, 256-token generation):

- Decode region: 186 887 kernels spanning 3 873 ms. 506 boundaries (0.27%) show any gap; their total
  is 6.86 ms, **0.177% of decode time** (0.037 µs per boundary on average).
- Attention-layer profile (layer 3 at 4k context, 16 448 kernels): positive gaps total 5.2 ms,
  **0.142%** of wall time.
- The largest gaps in the decode region are between `q4k_gemv_kernel → add_rmsnorm_kernel`,
  `moe_gate_up_kernel → moe_down_kernel`, `moe_down_kernel → add_rmsnorm_kernel` and
  `add_rmsnorm_kernel → moe_router_kernel`: these are the transitions in which one small launch
  hands over to the next, and they are microseconds in total.

Per-operator resources are also spread: decode operators range from 0 LDS / 20 VGPR
(`moe_combine`, `argmax`) to 42 KiB LDS (`attn_split`) and 256 VGPR with 66 spills
(`deltanet_seq`). A single-shape megakernel would inherit the maximum, so its occupancy would be
set by the fused attention and DeltaNet operators, not by the tiny ones it would be joining.

## Decision

**Do not adopt a persistent megakernel for decode.** The ADR_001 D4 trigger is measured false on
gfx906 as configured here:

- The inter-kernel overhead that a megakernel would remove is **0.18% of decode time** — about 14 µs
  in a 7.75 ms token, against a stage-3 budget of 5300 µs. It is not a material share.
- A grid barrier costs the same as a kernel boundary (1.81 vs 1.69 µs at 60 blocks), so even an
  aggressive fusion of every boundary into a barrier would not reduce that residual.

Decode keeps issuing fused per-operator kernels on a stream.

## Consequences

- Stage 3 must come from per-kernel work: the bandwidth floor is 379 tok/s and decode is at
  129 tok/s, so the remaining budget is in the kernels themselves (MoE 19.6% + 13.1%, dense GEMV
  19.5%, norms 10.5%, attention split/combine at long context), not in dispatch.
- The D4 prerequisites stay in force, so the option is not closed: operators remain `__device__`
  functions with explicit work ranges, all decode operators take the block size as a template
  parameter, and cross-operator dependencies remain explicit. Re-opening the megakernel decision
  needs a new measurement in which dependent-kernel gaps are a material share (the ~1.1 µs/step
  regime of probe B, not the hidden-dispatch regime of a full queue).
- `bench/mi50/grid_barrier.hip` is kept as the reusable probe for that re-measurement.
