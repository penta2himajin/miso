# DeltaNet register-only FP32 rounding boundary

Base `f258ce1` (PR36 merged), MI50/gfx906, ROCm6.3.4, Release/Ninja/C++20.
This continues the [previous preparation experiment](../2026-09-27-fp16-prepare/README.md).
Production sources are restored; native A/B used frozen prototype binaries.
All comparisons are complete. Retain the benchmark/test improvement; do not adopt
the current native prototype as a Decode optimization.
Exact original Red build output and live patch are retained as `.gz`; their text
renderings remove trailing whitespace. Selected audit ISA trims blank EOF lines.

## Observed numerical cause and implementation

The direct half cast again fails the guarded trajectory at layer0/grid32/token7
([Red](live-red.txt), [prototype patch](live-prototype.patch)). Actual emitted
[live producer ISA](live-step-3-isa.txt.gz) fuses the final FP32 multiplication and
half conversion into `v_fma_mixlo_f16` at0xe614. This omits the intermediate FP32
rounding performed by the baseline multiplication before its FP32 output store.

The register variant uses `asm volatile("" : "+v"(rounded))` before converting to
half. The opaque FP32 register boundary keeps that multiplication in FP32. It emits
no additional hardware instruction. [Measured producer ISA](micro-step-3-isa.txt.gz)
has `v_mul_f32` at0xb01c followed by `v_cvt_f16_f32` at0xb020, with no private final
store/load. [Resources](resources.txt):182VGPR/54SGPR/2576B LDS/0scratch, compared
with volatile Half-only182/62/2576/8B scratch. Occupancy ceiling stays1 wave/SIMD.
The half reader remains110VGPR/36SGPR/0LDS/0scratch, ceiling2 waves/SIMD.

The separate [compiler audit](audit-build.txt) compiles baseline, volatile, live
and register wrappers with the same flags. [Optimized LLVM](audit-optimized.ll.gz)
retains contracted `fmul float` then `fptrunc` for the live case; target lowering
produces the mixed FMA. [Selected ISA](audit-selected-isa.txt) corroborates the
separate multiplication/cast with the register barrier. Isolated raw SGPR counts
are not interchangeable with linked executable resource metadata. Exact pre-format
input is retained in `audit-source-compiled.hip.gz`; `audit-source.hip` is its
formatted rendering. `audit-reproduce.sh` compiles without running the GPU.

This resolves the need for a *memory* boundary in the previous prototype. It does
not prove a whole-model speed benefit. Compiler instruction selection must be
rechecked if this pinned compiler/architecture changes.

## Correctness

[Green test](green-test.txt):4 cases/4413 assertions, no skips. The continued
34-token test checks2 layers×2 grids (32/4), all state/conv bits, guarded rounded
outputs, GEMV output bits, and null FP32 output pointers for both half-only modes.
The register version is compared to an independent frozen FP32 producer plus cast.
All previous tests remain intact; no tolerance was relaxed.

[Check](check.txt):17-token chains and34-update batches, both layers, all6 variants
and3 scopes, bitwise and finite pass. Every timed64/2000-update batch is checked
outside its events. Frozen native prototype `test_model` and `test_deltanet` pass
([model](native-model-test.txt), [DeltaNet](native-deltanet-test.txt)).

Restored-production full build and [CTest](final-ctest.txt):22/22 pass,110.08s.
All retained final-logit dumps contain248320 finite FP32 values
([validation](logits-validation.json)). Source/binary manifests, formatting,
summary matrix/order checks and patch applicability were verified.

## Matched micro measurements

Each batch length:3 processes×7 rounds×2 layers×3 scopes×6 variants=756 rows.
The original5 variants are preserved; `half_register` is added. Warmup500, then
state/conv/parity reset and synchronization outside events; all timed batches
start at phase0 of the17 projected inputs. No reset launches are timed. One
nonblocking stream and matching launch/error-check policies are retained.
18-case forward/reverse/rotation order gives3/4 before/after comparisons. The
summary validates the observed order as well as the complete matrix.

2000 updates ([raw](bench.txt), [summary](summary.json), [telemetry](bench.txt.smi.jsonl)):

- Baseline chain medians34.998/35.011µs; FP32 clone control34.932/34.946µs.
- Register chain29.286/29.287µs for layers0/4; median paired latency changes
  **−16.317%/−16.374%** against baseline,21/21 faster per layer.
- Against simultaneously measured volatile Half-only: **−1.383/−1.434µs**,
  **−4.514%/−4.665%**,21/21 faster per layer.
- Producer versus volatile:−2.430/−1.703µs,21/21 faster. Register producer still
  costs+1.595%/+2.334% versus baseline. Separately timed scopes are not additive.
- Consumer-only Half-only/register call exactly the same half reader; their near-zero
  paired differences are a control measurement, not a new reader improvement.

64 updates ([raw](short-batches.txt), [summary](short-summary.json),
[telemetry](short-batches.txt.smi.jsonl)):

- Register chain29.837/29.832µs; paired **−17.752%/−17.807%** versus baseline.
- Against volatile:−1.235/−1.237µs,−3.965%/−3.911%,21/21 faster per layer.

Absolute figures are medians of process medians. Changes are medians of matched
within-process/round comparisons, not ratios of separately aggregated medians.

## Scope and limits

Same periodic cache-hot proxy as PR36: layer0 fixture is matched; layer4 reuses
layer0 input with layer4 weights. This is not a natural autoregressive state/cache
trajectory. Input projection is untimed; real recurrent updates occur inside events.
Every defined state/output is finite and matches reference after each batch.
Different64/2000 percentages cannot be attributed solely to state transients.
Telemetry records clocks/temperature/power across load/warmup/checks and timing;
samples are not aligned to individual event intervals. CPU builds are completed
before native timing; GPU jobs are serialized. GPU-use≥90% samples:2000-update
sclk1725MHz/mclk1000MHz, junction37..64°C, power102..166W;64-update
sclk1485..1725MHz/mclk1000MHz, junction48..60°C, power67..162W. Short-native samples
span sclk1485..1725MHz/mclk800..1000MHz, junction36..66°C, power33..252W,
including model loading/prefill. 4K/32K samples use the same clock ranges, with
junction40..67°C/45..80°C and power36..264W/34..267W respectively. These are not
phase-aligned Decode clock ranges.

The register version removes the volatile temporary's16KiB FP32 write and16KiB
read for4096 outputs. Unique producer output plus one unique consumer activation
read is16KiB for half versus32KiB for the FP32 path. This counts logical accesses,
not measured HBM traffic. It also removes SGPR/private-address dependencies while
preserving intermediate rounding; this micro experiment does not isolate each
hardware contribution.

## Native experiment

Seven alternating-order short pairs ([summary](native-summary.json),
[bitwise proof](native-proof.json)) are complete. Paired throughput changes are
−0.067%, +1.080%, +0.672%, −0.269%, +2.039%, −0.538%, −1.596%: median
**−0.067%**, range−1.596%..+2.039%, **3/7 faster**. Separate throughput medians
are baseline148.8 and register149.7tok/s; their ratio is not the paired statistic.
pp512 paired median−0.007%, range−0.408%..+0.104%. All513 generated IDs and the
last step's full-vocabulary FP32 logits match in all seven pairs. All fourteen
final dumps share one hash; [one compressed dump](native-final-logits.bin.gz) is
retained. Intermediate tokens' full logits are not compared.

Restored4K ([summary](context-summary.json), [proof](ctx4096-proof.json)):two
alternating-order pairs, baseline139.911→register139.514tok/s (separate medians),
paired median**−0.281%**, range−1.035%..+0.473%,1/2 faster. All65 IDs and final
logits match; each restored repeat matches its own warmup. 32K ([proof](ctx32768-proof.json)):baseline117.602→register117.128tok/s,
paired median**−0.402%**, range−0.907%..+0.103%,1/2 faster. The four context pairs
all match65 IDs and final logits bitwise; compressed final dumps are retained per
context. Each context process warms up one restored64-token sequence, then reports
the median of three measured copies, with state/KV restore outside events. Prompts
repeat the fixture to length4K/32K and are synthetic. This small study does not
establish universal regressions or identify a hardware cause. No consistent short-native improvement
has been demonstrated; do not adopt this prototype on the micro gain alone.
[Integration patch](native-register.patch) reuses the existing output allocation;
[output instrumentation](native-instrument.patch) is outside events. The candidate
uses the benchmark-local frozen operators for this measurement only. Production
adoption would require proper operator reuse and guarded dispatch, not this patch.
[Manifest](manifest.json) freezes binaries/source/patches. Original baseline engine
and executable hashes match the previous experiment exactly.

`run_native.py` uses fresh logs and compares actual generated-ID arrays and final
logits bytes before recording hashes. `summarize_native.py` summarizes only complete
proof pairs; confirm final counts before making a claim. Short throughput is printed
to0.1tok/s. The context probe shares one host object across the two engine archives.

## Reproduction

Build `bench_activation_prepare` and `test_deltanet_prepare`, then run the test and
`--check --iterations 34`. Use the existing `run_measure.py` with3 processes,
`--iterations 2000 --rounds 7 --warmup 500`, then repeat with64 and fresh output paths.
`summarize.py LOG SUMMARY` validates ordering and preserves every pair.
[Native freezing/reproduction](NATIVE_REPRO.md) gives exact patch, restore, build,
and measurement commands. Use fresh paths to preserve all existing raw logs.

## Decision and next diagnostic

The opaque register boundary solves the rounding counterexample without scratch
and reduces the periodic micro chain beyond the volatile control. Short-native
mixed signs and both context medians below zero do not support production adoption.
A fixed2% gate is not used. The two campaigns' volatile/register native numbers
are not a controlled causal comparison; their different runs cannot establish
scratch's whole-model contribution.

Next measure CPU enqueue walltime alongside whole-native GPU events, then sample
actual layer producer→GEMV chains (sparse events, rotating layers) with and without
instrumentation. CPU enqueue includes overlap/queue backpressure and is not a pure
launch cost. Timestamp traces can change the comparison itself; use a matched
uninstrumented control ([prior trace limitation](../../../norm/results/2026-09-27-latency/README.md)).
First determine whether the local chain win survives inside the full model before
attributing the result to host gaps, cache interference or weight traffic.

Any reprofile parser must recognize the prepared producer name and distinguish
DeltaNet's30 output GEMVs from the10 unchanged full-attention K4096 GEMVs. Re-derive
the measured dispatch range for each executable; do not reuse old profiler ranges.
