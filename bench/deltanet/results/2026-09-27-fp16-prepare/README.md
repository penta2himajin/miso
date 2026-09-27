# DeltaNet producer FP16 prepare → K4096 out GEMV

Base: `af54ec7` (main after MoE coefficient-once #35). MI50 32 GB, ROCm 6.3.4,
Release, gfx906. **Measurement milestone; production sources unchanged.**
The implementation is benchmark-local, with a temporary Half-only native integration
for the new full-model comparison described below.

## Hypothesis and controls

The Q4_K K=4096, N=2048 out-GEMV rounds the same 4096 FP32 activations in each
active wave. With R4 and grid 240, there are 512 active waves, not 960: the
remaining waves exit before activation loads (see [baseline ISA](baseline-gemv-isa-selected.txt)).
Preparing halves once in the producer removes redundant consumer casts and halves
the activation load width. The R4 row/weight pipeline and slot reduction order stay
unchanged. [Model header verification](model-shapes.json) confirms all30 DeltaNet
output tensors are Q4_K with dimensions4096×2048. This is a hypothesis about the complete chain, not a measured HBM bound.

- `baseline`: production FP32 step and production FP32-input GEMV.
- `float_control`: frozen step clone with FP32 output and the same GEMV.
- `dual_float`: FP32 plus prepared FP16 output, followed by the FP32-input GEMV.
- `dual_half`: the same Dual producer, followed by the FP16-input GEMV.
- `half_only`: prepared FP16 output only, followed by the FP16-input GEMV.

## Numerical boundary and producer cost

The recurrence and convolution state remain FP32. FP16 conversion follows the final
gated RMSNorm output expression, outside the recurrence. Dual writes through a
volatile FP32 pointer and reloads that word before casting. Half-only uses a volatile
local FP32 temporary. A reconstructed live-register cast fails the trajectory test
at step 7; [the patch](live-cast-red.patch) and [Red log](live-cast-red.txt) preserve
the counterexample. The observed mismatch does not establish its compiler-level cause.

The volatile temporary is not free register arithmetic. [Current resources](bench-resources.txt)
show producer VGPR 182 and LDS 2576 B in all modes; baseline/control use SGPR 54,
Dual 56, Half-only 62. Half-only has 8 B private scratch per work-item, with zero
reported spills; the compiled occupancy ceiling remains 1 wave/SIMD. The native
Half-only build has [the same allocation](native-half-resources.txt). For 32×256
launched threads this is 64 KiB of logical private allocation. Both GEMV readers
remain VGPR 110, SGPR 36, LDS/scratch zero, ceiling 2 waves/SIMD.

Dual includes a FP32 store/reload and FP16 store. Half-only replaces the FP32 global
output with a private FP32 store/reload and a FP16 global store. Counting the final
4096-element materialization and one unique consumer activation read gives baseline
32 KiB versus 48 KiB for either prepared chain, including the private accesses in
Half-only. These are logical access bytes, not measured L1/L2/HBM traffic. In particular,
the earlier no-roundtrip argument that Dual keeps unique traffic at 32 KiB does not
apply to this implementation. The experiment measures the additional accesses and
dependencies together with reduced consumer conversion/load work.

## Correctness

- [Final Green](final-test.txt): four test cases, 3053 assertions. The 34-step
  trajectory test checks layers 0/4 with grids 32/4, including every step's state,
  both convolution histories, guarded outputs, rounded halves and output GEMV bits.
  Half-only also runs with a null FP32 output pointer. Comparisons are bitwise;
  tolerances are not relaxed.
- [Check run](check-run.txt): consecutive 17-token chains and 34-update batches,
  both layers, all variants/scopes, bitwise and finite checks pass.
- Every timed 64/2000-update batch is checked outside the events against the
  production reference for the outputs/state applicable to its scope. Both logs
  contain three successful process exits.
- The current temporary Half-only native integration passes `test_deltanet` and
  `test_model` ([raw test log](native-half-tests.txt)). Full-sequence native comparisons
  pass for all seven short native pairs and all four restored-context pairs.
  These tests alone do not prove equality for all possible inputs.

## Micro protocol and results

Each batch length has 3 processes × 7 rounds × 2 layers × 3 scopes × 5 variants:
630 rows. Warmup is 500 updates. Each warmup starts from zero; state, both conv
buffers and parity are reset again and synchronized before recording the start event.
Timed updates begin at phase zero and rotate through 17 actual projected fixture
inputs. There are no reset launches inside the events. Forward/reverse order alternates
with a rotated start. Producer costs and per-launch host error checks are included
symmetrically. Final correctness checks occur after the end event.

The figures below use **median paired latency changes** within the same process,
layer, scope and round. Negative values mean faster. Absolute latencies are separately
the median of the three process medians; the percentage is not a ratio of those medians.

**2000 updates**: [raw log](bench.txt), [paired summary](summary.json),
[telemetry](bench.txt.smi.jsonl).

- Baseline chain: layer 0/4 = 35.027/35.066 µs; FP32 clone control = 34.956/34.974 µs.
- `float_control` versus baseline: −0.203%/−0.237%.
- `dual_float` versus baseline: +1.894%/+1.783%; versus control: +0.766/+0.703 µs.
- `dual_half` versus baseline: −11.007%/−11.247% (−3.859/−3.944 µs);
  versus control: −3.770/−3.865 µs, each layer 21/21 faster pairs.
- `half_only` versus baseline: −12.256%/−12.258% (−4.295/−4.299 µs);
  versus control: −4.224/−4.210 µs, each layer 21/21 faster pairs.

**64 updates**: [raw log](short-batches.txt), [paired summary](short-summary.json),
[telemetry](short-batches.txt.smi.jsonl).

- Baseline chain: layer 0/4 = 36.402/36.537 µs; FP32 clone control = 35.702/35.672 µs.
- `float_control` versus baseline: −1.158%/−2.227%.
- `dual_float` versus baseline: +0.663%/−0.323%; versus control: +0.752/+0.687 µs.
- `dual_half` versus baseline: −13.366%/−14.136% (−4.837/−5.165 µs);
  versus control: −4.245/−4.323 µs, each layer 21/21 faster pairs.
- `half_only` versus baseline: −13.448%/−14.909% (−4.880/−5.447 µs);
  versus control: −4.527/−4.627 µs, each layer 21/21 faster pairs.

The improvement remains against the FP32 clone control, so the baseline/clone
difference alone cannot explain it. Consumer-only improves approximately 30–32%,
but its halves come from an untimed production/cast cache; it excludes preparation
cost and is diagnostic. Producer-only prepared modes cost more than baseline.
Separately measured scopes are not additive measurements of the complete chain.

## Scope and limitations

These are periodic, cache-hot, single-layer proxies. The 17 input projections are
prepared before timing; the model's natural autoregressive input/state trajectory,
other layers' weight traffic and their cache interference are absent. Layer 0 uses
its matched fixture. Layer 4 reuses `layer0.in` (`matched_fixture=0`), with layer 4's
actual weights; it is not a natural layer 4 input fixture.

Final FP32 state has no subnormals in either batch length. Final prepared halves
do contain subnormals (roughly 3.4–5.8%); neither output vector is all zero.
The lag-17 output relative L2 change is 4.29e−5/5.29e−5 at 2000 updates and
0.00993/0.06723 at 64 updates for layer 0/4. Thus the long batch is close to a
periodic steady output, while the short batch contains larger transients. This
output comparison is not a full-state stationarity proof. The two runs also differ
in time/temperature; their percentage difference cannot be attributed solely to state.

Logged GPU-use≥90% samples show sclk 1725 MHz and mclk 1000 MHz in both runs.
Junction temperatures span 51–65°C for 2000 and 40–52°C for 64. Telemetry includes
load, setup, warmup and checks and is not phase-aligned. Event times include any
submission-induced GPU gaps; CPU enqueue time can include queue backpressure.
Neither per-layer micro percentages nor separately timed scope deltas establish a
full-model Decode gain.

## Earlier Dual native logs: provisional

The three older `e2e-*-baseline.txt`/`e2e-*-candidate.txt` pairs report short Decode
147.5→149.0, 149.5→148.4 and 150.3→147.2 tok/s (+1.02%, −0.74%, −2.06%).
The original candidate source patch and run-specific binary archives were not
preserved. [Resume manifest](resume-manifest.json) records the interrupted artifacts
and their hashes, including a stale temporary Dual executable and a baseline engine
archive. These logs remain **provisional** and do not establish the current Half-only
candidate's native behavior or a cause for the old mixed results.

The previous “≥2% gate” and causal **REJECT** conclusion are withdrawn. That gate
was not specified for this experiment; “Dual cost eats most of the full-model win”
was not demonstrated by those logs. No universal gain or regression is inferred.

## New Half-only native comparison

Seven alternating-order pairs use the current volatile-temp Half-only producer and
FP16 reader, reusing the existing output allocation. [Integration](native-half-only.patch)
and [outside-event output instrumentation](native-instrument.patch) are archived.
Both binaries share the instrumentation; [manifest](final-manifest.json) preserves
source and executable hashes. Restored production `bench_decode` exactly matches
the original `afe4f824...` baseline SHA256.

[Raw native logs](native-1-baseline.txt), [paired summary](native-summary.json),
[output proof](native-proof.json), and all fourteen telemetry sidecars are committed.
Measured Decode is 256 tokens at ctx273..529 after 256 warmup tokens. Paired tok/s
changes in chronological order are +1.273%, +1.343%, +0.949%, −0.799%, −0.068%,
+0.405%, −0.668%: median **+0.405%**, range −0.799%..+1.343%, **4/7 faster**.
Separate baseline/candidate throughput medians are both 148.9 tok/s. pp512 paired
median is 0.000%, range −0.166%..+0.160%.
The short benchmark prints throughput to0.1tok/s; these percentage calculations
use those reported rates rather than unrecorded higher precision timings.

All 513 greedy prediction IDs and the final step's full-vocabulary FP32 logits
are bitwise equal in every pair. All fourteen final logits dumps have one SHA256;
one [compressed dump](native-final-logits.bin.gz) is retained for verification.
This does not compare all intermediate tokens' full logits or establish correctness
for arbitrary prompts. [Emitted native ISA](native-half-isa-manifest.json) contains
a private FP32 store/load and one producer conversion; half GEMV has zero FP32→FP16
conversions. The original GEMV has 64 static conversions in its loop body.

GPU-use≥90% samples span sclk1485..1725MHz, mclk800..1000MHz, junction37..66°C,
and package power32..257W. These samples include model loading, prefill, warmup and
Decode and are not phase-aligned, so they do not identify the cause of the timing
variation. The first baseline process overlapped the CPU rebuild of restored main;
excluding that pair leaves +0.169% paired median and3/6 faster ([provenance](final-manifest.json)).
There is no consistent short-native improvement demonstrated by these seven pairs. Restored-context measurements reinforce the decision below.

## Restored-context4K and32K

[Probe source](context_probe.hip), [context summary](context-summary.json),
[4K proof](ctx4096-proof.json), [32K proof](ctx32768-proof.json) and all eight raw
logs/telemetry sidecars are retained. Each context has two alternating-order pairs.
A prompt repeats the17-token fixture to the required length; it is synthetic.
Each process prefills once, warms up one restored64-token Decode trajectory and
measures three copies of that same trajectory, restoring recurrent/KV state and
last prediction outside events. Each process reports the median of its three
rounds, then the summary computes paired baseline/candidate throughput changes.

- 4K: baseline140.787→Half-only138.993tok/s (separate medians); paired median
  **−1.274%**, range−1.470%..−1.078%, **0/2 faster**.
- 32K:117.327→117.340tok/s; paired median **+0.013%**,
  range−0.455%..+0.480%, **1/2 faster**.

Every restored repeat matches its own warmup's64 IDs and final logits bitwise.
Cross-binary65 IDs, including the initial prediction, and final logits match in
all four pairs. One compressed logits dump per context is retained. This is a
small, fixed-trajectory study; it does not establish universal regressions or
the cause of the4K slowdown. Together with the short-native mixed signs, it does
not support adopting this scratch-producing prototype as a Decode improvement.

## Current decision

Keep the measurement harness and counterexample tests; do not integrate this
scratch-producing prototype into production based on the micro percentages. The4K regression and32K flat result give no
reason to change that decision.
A future register-only FP32 rounding boundary may be worth measuring, but it must
retain the token7 counterexample and complete state/output bitwise checks. Neither
scratch causality nor a gain from removing it has been demonstrated here.

## Final verification

[Restored-main build](final-build.txt) and [full CTest](final-ctest.txt):22/22 pass
in102.30s. Production engine/benchmark sources are restored, baseline binary SHA256
unchanged. Formatting, Python compilation, patch applicability, data/proof and
source/binary hash checks pass.

## Reproduction

Use new output paths when repeating measurements; preserve these raw logs.
[Native reproduction instructions](NATIVE_REPRO.md) include both temporary patches,
source restoration, frozen binaries and the restored-context probe.

```bash
cmake --preset default && cmake --build --preset default \
  --target bench_activation_prepare test_deltanet_prepare
./build/tests/test_deltanet_prepare
./build/bench/bench_activation_prepare --check --iterations 34
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py \
  /tmp/prepare-repeat.txt 3 \
  ./build/bench/bench_activation_prepare --iterations 2000 --rounds 7 --warmup 500
python3 bench/deltanet/results/2026-09-27-fp16-prepare/summarize.py \
  /tmp/prepare-repeat.txt /tmp/prepare-repeat-summary.json
# Repeat with --iterations 64 and separate output paths for the short batch.
```
