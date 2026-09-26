# Decode attention combine: prepare coefficients, then prefetch outputs

Base: `6d10fb5`; MI50/gfx906, ROCm 6.3.4, default clocks/power profile.
The post-KEEP trace attributes 83.7 us/token (short) and 268 us/token (32k)
to ten combine calls. The original kernel has 16 256-thread workgroups.

## Hypotheses and isolated results

1. **Dimension tiling alone — REJECT.** Four 64-dimension tiles/head give
   64 one-wave workgroups. This exposes more CUs but leaves serial load waits.
   At 60 splits: 27.23 -> 27.30 us; essentially flat.
2. **Parallel coefficients.** One lane owns each split's max/denominator;
   a wave max produces the same maximum, then each weight is computed once.
   Broadcasts retain the original split-ordered FP32 denominator/output sums.
   At 60 splits: 27.25 -> 19.82 us. The first short e2e batch was inconclusive
   (145.9 -> 144.4 unpaired); this intermediate version is not production.
3. **Four output loads before accumulation.** Pipeline four consecutive
   `part_o` reads, then perform the original ordered FMAs. At 60 splits:
   **27.25 -> 8.14 us (-70.2%)**; at 16 splits: 8.83 -> 4.94 us (-44.0%).
   This is the production candidate, measured separately and with mixed GEMV.

`micro-tiled.txt`, `micro-prepared.txt`, `micro-pipeline.txt` contain seven
rotating-order samples per variant per split count, each 300 warm-ups followed
by 2000 launches. Each matching telemetry log includes clocks/temperature/power.
These are warm-cache microbenchmarks; native model measurements decide adoption.
The ns=1 setup is slightly slower (3.80 -> 3.97 us); restored short-context
model runs must therefore be included. Above 64 splits, the original algorithm
is used. Production caps splits at 60, while tests exercise 63/64/65/129.

## Correctness and ISA

TDD Red/Green logs cover the dimension tile, prepared coefficient and pipeline
variants. The final test adds 516096 bitwise comparisons against a frozen copy
of the original combine. Cases include split-count/tile boundaries, empty
partials, large score differences, extreme gates, partial ranges and grid stride.
The existing FP64 split-attention tests remain, including 32k +/-1 boundaries.

`pipeline-combine-isa.txt` is the selected function from the full disassembly
(`pipeline-isa.txt.gz`). At offsets 0x62a0/0x62b8/0x62c0/0x62c8, all four
output loads precede the output FMAs at 0x636c/0x6374/0x637c/0x6384; waits
retire them in order. This is the evidence for the prefetch claim. Resources:
25 VGPR, 43 SGPR, no LDS/scratch/spills. The reported occupancy ceiling is 9
waves/SIMD, not a measurement of achieved occupancy.

Model-wide measurements and decisions are in
`../../../model/results/2026-09-27-first-principles/`. `bench_combine` is the
repeatable isolated benchmark target. Build with Ninja and run under the
existing `run_measure.py` harness. The raw logs identify the exact commands.
