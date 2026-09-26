# REJECT: Q4 K=4096 GEMV R=4→2 (occupancy)

Base: post-KEEP re-profile counters (`bench/model/results/2026-09-27-reprofile/`).
`q4k_gemv` K=4096 R=4 shows VALUBusy ~14% / occ 2 under short decode. Hypothesis:
R=2 cuts VGPR (110→78) and raises occ 2→3 for latency hiding. Distinct from the
rejected M7c R=5/6 and small-grid trials (those went the other way or shrank the grid).

## Change (reverted)

Only `gemv()` dispatch for `w.k == 4096`: R=4 → R=2 at the same 240-WG grid.
`test_gemv_dispatch` and `test_model` passed on the candidate.

## Paired results (median of 3)

| probe | baseline | candidate | change |
|-------|--------:|----------:|-------:|
| short e2e | 144.0 | 144.0 | **0.00%** |
| pp512 | 1440.3 | 1439.8 | −0.03% |
| ctx 17 | 145.677 | 144.617 | −0.73% |
| ctx 4096 | 133.493 | 134.970 | +1.11% |
| ctx 32768 | 112.582 | 112.060 | −0.46% |

## Decision

**REJECT.** Occupancy 3 does not move short e2e; gate ≥2% not met. Production stays R=4.
