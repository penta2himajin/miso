# Decode Norm latency controls (2026-09-27)

Measurement-only work from main `10a00da` after PR #33. No production kernel or
inference dispatch was changed. `manifest.json` identifies binaries, device code
objects and final benchmark sources. The production decode executable still has
SHA256 `2b562fab4ab452c806a03dbeb17f5509700f7ad701102a3e91d3388afdb101d6`.

## Finding

The available evidence does **not** make reduction/rsqrt tuning the priority.
In representative real-weight native chains, removing the Norm reduction while
keeping its logical reads, residual update and output writes changes the batch
by only **0.180 / 0.330 / 0.340 us per step** (GEMV / MoE-Q4 / MoE-Q6 chains).
The read/write control versus the no-op middle changes it by approximately
**5.855 / 6.040 / 6.057 us**. These are paired control increments, not exact
arithmetic costs or realizable fusion savings. Some individual rounds are
negative; distributions are retained in `measurements.json.gz`.

Timestamp tracing changes the control comparison itself. It cannot be calibrated
by subtracting one empty-kernel duration. Native whole-chain HIP-event measurements
are the primary evidence; neither the previous 685 us/token sum nor any new
control delta is an inference speedup bound.

## Protocol and scope

GPU was idle at the first check (0% use, junction 35 C); all GPU jobs were serial.
Default DVFS, no clock locking. Every reported series has raw stdout and
`.smi.jsonl` clock/memory-clock/temperature/power/use samples (ADR_002 D12).
Warmups, initialization, graph capture/instantiation and model-weight loading are
excluded from batch event timing. An explicit nonblocking stream is used for both
native and graph execution; events bracket batches, never individual kernels.

Primary numbers are the median of three outer-process medians, each with seven
interleaved rounds. Control differences are subtracted within a process and round
before aggregation. CPU enqueue wall time is also recorded; it overlaps GPU work
and is not added to device time. Graph captures the entire batch to reduce host
supply differences. All rounds and process medians are preserved.

`bench_norm` covers N=2048, block=512, plain and residual-add cases:
empty dispatch, copy, read/write/multiply without reduction, and production Norm.
Dependent execution uses genuine A/B ping-pong. Independent execution uses a
64-slot ring without adjacent RAW dependencies, with finite buffer reuse.
A second path sandwiches controls between one-workgroup vector producer/consumer
operators; Direct omits the middle dispatch. Empty carries no memory dependency.

`bench_norm_chain` loads actual model weights and production operators, with
synthetic inputs and fixed selected expert IDs:

- `gemv_norm_router`: layer0 K4096/N2048 SSM output GEMV, post-Norm, layer0 router.
- `moe6_norm_qkv`: layer0 Q6 MoE down, next input Norm, layer1 Q6 qkv GEMV.
- `moe4_norm_qkv`: layer5 Q4 MoE down, next input Norm, layer6 Q4 qkv GEMV.

The MoE probes use the selected 8192-row qkv matrix. They omit production's z/a/b
projection and its concatenated/mixed dispatch. They are representative real
operator chains, **not complete production decode**. Their repeated working sets
also differ from the model-wide cache/traffic workload. Direct/Empty consumers
read the producer's delta output; ReadWrite/Norm consumers read the middle output.
The controls change math, pointers and dependency behavior and are not inference
alternatives. Do not multiply their deltas by 81 to predict tok/s.

Telemetry summaries cover whole processes including loading, warmup and gaps;
they cannot assign clocks or power to individual variants. Controls reach sclk
1725 MHz but mclk only 800 MHz; real-weight chains reach mclk1000 MHz. Cross-series
comparisons therefore also include different hardware/cache states.

## Native and graph results

Residual-add standalone, dependent path, native medians:
empty1.253, copy3.948, readwrite5.181, Norm5.404 us/step.
The no-reduction control has much of the measured Norm cost.
Independent-ring Norm5.568 versus dependent5.404 does not support a large
adjacent-RAW penalty in these one-workgroup fixtures.

Real-weight native whole-chain medians (us/step):

- GEMV: direct25.905, empty26.138, readwrite31.993, Norm32.173.
- MoE-Q4: direct44.779, empty44.826, readwrite50.857, Norm51.206.
- MoE-Q6: direct50.968, empty51.003, readwrite57.049, Norm57.361.

Graph Norm-minus-readwrite paired increments are0.154/0.352/0.358 us. Graphs
reduce CPU enqueue work but give no material device-time improvement in these
chains; several graph variants are slightly slower. Native enqueue is about
0.9 us for standalone Norm versus its5.4 us device batch average, and about3 us
for three-kernel real chains versus32..57 us/step. This does not support host
supply as the principal explanation in the measured cases.

## Trace perturbation and predecessor correlation

The existing final #33 trace contains20736 Norm calls over256 tokens,81/token,
685.109 us/token total. Grouping the same kernel by predecessor gives median
5.60 us after embedding,6.88 after Q4 output GEMV,10.40 after MoE-Q4 down and10.72
after MoE-Q6 down. This is correlation, not proof of one memory/dispatch cause.
`summarize_norm_predecessor.py` checks token boundaries and matches prior totals.

Three matched native/traced process pairs use128 iterations,3 rounds,128 warmup;
pairs1/3 run native first, pair2 traced first. For the GEMV chain, paired
empty-minus-direct changes from **0.163 us native to5.032 us traced**, while
readwrite-minus-empty changes from **5.966 us native to0.595 us traced**.
MoE controls show the same qualitative shift. The timestamp mechanism changes
scheduling/synchronization or where time is accounted; this experiment does not
identify the exact mechanism. Native Norm-minus-readwrite stays small.
The old685 us/token remains a useful instrumented family total, not removable
execution cost. Tiny positive gaps between timestamps do not resolve this issue.

All matched raw CSV/stats/sysinfo files are gzip-preserved. Profiler stdout
including carriage-return progress is preserved as `.txt.gz`; readable `.txt`
copies normalize progress line endings and trailing whitespace. Initial exploratory
`profile-chain.*` and `native-chain-matched.txt` are also retained, but the three
matched pairs are the stronger comparison. `profile-chain-kernels.json` describes
the initial trace pool, including warmups and checks, not a decode-token window.

## ISA and checks

Red: missing read/write control fails the targeted build (`red-build.txt`);
initial real-chain read/write stub fails its residual/output oracle
(`red-chain.txt`). Green: standalone and all vector ping-pong/ring controls pass
CPU oracles, including explicit-stream graphs (`validation-check.txt`).
Real-weight native/graph outputs and residuals are bitwise equal after four
steps for every control (`green-chain-check.txt`). Plain/residual Norm is checked
against a high-precision CPU oracle; there is no tolerance change in production.

Review found `--slots 1` would contradict independent-ring metadata; a Red/Green
check now rejects it (`red-slots.txt`, `green-slots.txt`). This host validation was
added after measurement; final gfx906 code objects are byte-identical to the
measured ones (`manifest.json`). Both targets build; production files are unchanged,
so no full model CTest campaign was repeated for this benchmark-only change.

Archived ISA/resource reports show full Norm26 VGPR/32 B LDS, readwrite11 VGPR/no
LDS, no scratch in either. Norm has two `s_barrier` instructions at0x3030/0x3098,
`v_rsq_f32` at0x3090, and late weight loads/waits, e.g.0x30bc/0x30c4, in
`norm-isa-selected.txt`. The read/write control has different resources and load
ordering; this reinforces why its difference is not pure arithmetic subtraction.
Occupancy figures are compiled ceilings, not achieved occupancy of a one-WG grid.

## Next experiments

1. **MoE top-k coefficient redundancy**: gate/up currently computes coefficients
   in all120 workgroups, although only block0 exports weights for down. Preserve
   replicated ID selection and its barrier; compute softmax coefficients only in
   block0. The existing gate/up profile budget is1161.8 us/token (18.01%), but the
   redundant setup portion is unmeasured. Production ISA confirms coefficient
   `v_exp_f32` at0x6b18..0x6e30 before the first block-zero comparison at0x6ec8
   (`moe-coefficient-isa-selected.txt`); the compiler has not sunk these into block0.
   Start with exact coefficient/ID tests,
   setup and complete gate/up A/B, then native decode. No extra launch/global barrier.
2. **K4096 activation preparation in its producer**: output GEMVs consume834.9
   us/token (12.94%) in the existing profile. Test producing FP16 activations once
   from DeltaNet/attention output while preserving rounding and offset-sum order,
   and consuming them with unchanged weight layout/R=4/grid. Include producer
   store/register overhead. Previous K2048 Norm FP16 preparation was rejected;
   this is a different producer/K4096 experiment, with uncertain benefit.
3. **DeltaNet LDS lifetime separation**: step has383.4 us/token (5.94%). Separate
   kv and output partial arrays (+1024 B LDS) to remove one overwrite-protection
   barrier without changing FP32 arithmetic order. Eight barriers and small total
   budget make this a lower-priority micro experiment, not a large-gain claim.

A new Norm fusion remains possible, but must preserve residual ownership and
explicit synchronization and be tested as a complete producer/consumer chain.
Do not repeat prior redundant consumer Norm, residual-only producer fusion,
float4 Norm, or block128/256 trials without a distinct mechanism. Reducing182
DeltaNet VGPR alone does not create more than its32 workgroups on60 CUs.
The old `grid_barrier` probe does not form a genuine K-step RAW chain and has
multiple writers; its timings cannot calibrate the new Norm hypothesis.

## Reproduction

```bash
cmake --preset default
cmake --build --preset default --target bench_norm bench_norm_chain
build/bench/bench_norm --check
build/bench/bench_norm_chain --check
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py \
  bench/norm/results/2026-09-27-latency/controls.txt 3 \
  build/bench/bench_norm --mode both --iterations 2048 --rounds 7 --warmup 512
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py \
  bench/norm/results/2026-09-27-latency/model-chain.txt 3 \
  build/bench/bench_norm_chain --iterations 512 --rounds 7 --warmup 512
python3 bench/norm/results/2026-09-27-latency/run_trace_pairs.py
python3 bench/norm/results/2026-09-27-latency/summarize_measurements.py
python3 bench/norm/results/2026-09-27-latency/summarize_norm_predecessor.py
```
