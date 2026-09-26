# Independent mixed-Q4/Q6 decode projections in one grid

Base: `6d10fb5`; MI50 gfx906, ROCm 6.3.4, Release, default power profile.

## Hypothesis

Mixed-type projections cannot use the existing same-type row concatenation.
14 DeltaNet layers need Q6 qkv + Q4 z/alpha/beta, and six attention layers need
Q4 q/k + Q6 v. They share an input and have disjoint outputs. Interleave their
original operators in one grid on the caller's stream. This removes 20 kernel
executions/token and permits scheduling both work types together. It is a
separate design from the rejected non-blocking side-stream implementation.

The original logical grids and per-row arithmetic remain: DeltaNet uses
Q6 R=2/grid480 and Q4 R=4/grid240; attention uses Q6 R=1/grid240 and Q4
R=4/grid240. Block stripes contain 2:1 or 1:1 Q6:Q4 workgroups. Generic shapes
fall back to the original two GEMVs. No events, side stream or additional
persistent scratch are needed.

## Validation and resources

Red/Green compiler logs cover the kernel and host API. `test-gemv.txt` records
172611 passing assertions: real full/odd-row matrices, canaries, reversed API
arguments, null and non-blocking streams, and bitwise output equality. The
existing model continuation test passes (`test-model.txt`).

The ISA files show the uniform branch and unchanged dot/reduction construction.
DeltaNet mixed uses 63 VGPR/43 SGPR (ceiling4), attention mixed 68/40 (ceiling3),
both zero LDS/scratch/spills. Standalone Q4 R=4 uses 69 VGPR/ceiling3; Q6 R=2
uses 60/ceiling4. The mixed compiler's register allocation differs, so an e2e
win alone cannot distinguish reduced fixed execution cost, occupancy, and
latency overlap. Resource ceilings are not achieved-occupancy measurements.

## Native comparisons

Three interleaved short model runs: baseline 145.1/146.9/146.0 (median146.0),
mixed-only 148.8/147.8/148.8 (median148.8, +1.92%). All three matched-round
comparisons are positive; their gains range from +0.61% to +2.55%.
The production decision is based on the combination with attention combine
at all required contexts, recorded in
`../../../model/results/2026-09-27-first-principles/`.
