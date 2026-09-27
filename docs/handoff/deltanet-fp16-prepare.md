# Handoff: deltanet-fp16-prepare

## Snapshot

- Branch: `ai-written/deltanet-register-rounding`
- Base: `f258ce1` (PR36 merged)
- Last work commit: `daa8572` @2026-09-28
- Working tree: clean after this handoff commit; production sources equal main
- Last session: 2026-09-28 00:42 JST

## Status

ready-for-review: register-boundary measurement milestone complete; production
adoption declined. PR36 merged into main at `f258ce1`. This branch retains only
benchmark/test changes and immutable measurement results.
[PR37](https://github.com/penta2himajin/miso/pull/37) is OPEN, ready and MERGEABLE,
and attached to this chat. Auto-subscription was attempted immediately after creation;
GitHub returned `INSUFFICIENT_SCOPES` because the token lacks `notifications`.
No authentication settings were changed.

## Next action

Review PR37, then measure native CPU enqueue walltime alongside GPU
events and sparsely sample actual DeltaNet producer→GEMV chains, rotating layers.
Use matched uninstrumented runs to quantify measurement perturbation. First establish
whether the local chain gain survives inside the full model; do not attribute the
whole-model outcome to CPU gaps, cache interference or HBM without further evidence.

## Verification

- Four `test_deltanet_prepare` cases /4413 assertions pass, no skips. Direct cast
  Red reproduces at layer0/grid32/token7; register-boundary Green preserves all
  state/conv, guarded rounded output and GEMV bits on2 layers×2 grids×34 tokens.
- Both2000/64-update micro matrices:756 rows each,3 processes×7 rounds,
  all state/output bitwise and finite checks pass. Observed order validates3/4
  before/after comparisons for every candidate/scope.
- Frozen register prototype `test_deltanet`/`test_model` pass.
- Short native7 pairs: all513 IDs and final full-vocabulary logits match bitwise.
  Throughput paired median−0.067%,3/7 faster, range−1.596%..+2.039%.
- 4K/32K restored-context2 pairs each: all65 IDs and final logits match; restored
  repeats match their warmup bits. Paired medians−0.281%/−0.402%,1/2 faster each.
  Intermediate full-vocabulary logits are not compared.
- Restored-production full build andCTest:22/22 pass,110.08s. Source/binary/audit
  hashes, format/diff/Python/shell, matrix/order and patch checks pass.
- Restored `bench_decode` SHA256 equals original main `afe4f824...`; engine archive
  equals `b09779bc...`. Production sources have no diff. All CPU builds finished
  before native timing; GPU jobs serialized.

## Context pointers

- Current results: `bench/deltanet/results/2026-09-28-fp16-register/README.md`
- Native reproduction: same directory `NATIVE_REPRO.md`, patches, manifests,
  emitted ISA/resources, raw telemetry, final-logit dumps and `SHA256SUMS`.
- Prior immutable results: `bench/deltanet/results/2026-09-27-fp16-prepare/README.md`
- Benchmark-local operators: `bench/deltanet/activation_prepare.hpp`
- Matched reset/order/checks: `bench/deltanet/bench_activation_prepare.hip`
- Guarded trajectory and token7 counterexample: `tests/test_deltanet_prepare.hip`

## Decisions made

- Direct half cast contracts FP32 multiply and truncation into `v_fma_mixlo_f16`,
  skipping baseline intermediate FP32 rounding (live linked ISA at0xe614).
  The opaque FP32 register boundary preserves `v_mul_f32`→`v_cvt_f16_f32`
  (measured ISA at0xb01c/0xb020), emits no extra instruction and removes scratch.
  Memory storage is not required to preserve rounding with this pinned compiler.
- Register producer:182VGPR/54SGPR/2576B LDS/0scratch versus volatile
  Half-only182/62/2576/8B. Occupancy ceiling remains1 wave/SIMD. Recheck ISA if
  compiler/architecture changes; isolated raw SGPR counts differ from linked metadata.
- 2000-update periodic chain: paired latency−16.317%/−16.374% versus baseline,
  −4.514%/−4.665% versus simultaneous volatile Half-only;21/21 faster per layer.
  64-update sensitivity also improves. These local gains do not establish native gains.
- No production adoption: short-native signs are mixed and both context medians
  are below zero. No fixed2% gate or causal whole-model regression claim is used.
  Separate throughput medians' ratio is not the paired statistic.
- Layer4 reuses layer0 input with layer4 weights. Periodic cache-hot micro inputs
  and synthetic context prompts limit scope. Telemetry includes load/prefill/warmup
  and is not aligned to Decode events. Prior and current native campaigns are not
  a matched comparison of volatile versus register scratch costs.
- CPU enqueue includes overlap/queue backpressure; it is not pure launch cost.
  Reprofile parsers must recognize the new prepared producer and distinguish30
  DeltaNet output GEMVs from10 unchanged full-attention K4096 GEMVs. Re-derive
  current dispatch ranges; old profiling ranges are not reusable.

## Failed approaches

- Direct live half cast: bitwise trajectory fails at token7. Patch/log/full ISA
  archived; register boundary fixes the numerical cause without private store/load.
- Volatile Half-only (PR36): local gain, mixed native gains; current register
  experiment resolves scratch but still shows no consistent whole-model gain.
- Current register native prototype: local chain improves but7 short pairs and
  2 pairs per context do not support adoption. Temporary engine wiring is archived
  for reproduction and restored; production reuse/guarded dispatch is not implemented.
- Legacy Dual3-pair study remains provisional because exact integration archives
  are missing. Earlier unsupported causal REJECT and2% gate are withdrawn.

## Session log

- 2026-09-27: Took over Codex WIP (bench green, test link incomplete). Fixed tests,
  volatile store roundtrip, measured micro (−11% dual_half chain) and short e2e
  (−0.74% paired median). REJECT; production reverted; harness + results kept.

- 2026-09-27 (resumed Codex): Preserved prior data and stale build hashes. Rebuilt
  baseline to the exact original SHA256, reproduced live-cast Red at token7, and
  extended state/conv/core/GEMV/canary checks to34 tokens on layers0/4 and grids32/4.
  Measured64-update sensitivity without overwriting2000-update logs. Archived
  Half-only native wiring and outside-event ID/logit dump patches. Candidate
  model tests and seven native pairs remain to finish; production sources restored.

- 2026-09-28 (completed measurement milestone): Retained legacy Dual logs as
  provisional and withdrew the unsupported2% gate/causal REJECT explanation. Added
  real34-token exact tests and live-cast Red,64-update sensitivity, seven Half-only
  native pairs plus two pairs each at4K/32K. All generated-ID/final-logit comparisons
  passed. Short-native paired+0.405% (4/7),4K−1.274% (0/2),32K+0.013% (1/2):
  decline the current prototype, without attributing mixed timings to scratch.
  Archived exact pre-format context source, ISA/resources/patches/hashes and raw
  telemetry, verified all30 target matrix shapes, restored exact main binary and
  passed22/22 CTest. Work commit `cee421d`; measurement PR delivery follows.

- 2026-09-28 (delivery): Opened ready PR36, attached it, confirmed mergeability,
  and attempted auto-subscription. Notification scope is unavailable; delivery and
  results are complete, with a clean pushed branch.

- 2026-09-28 (register-boundary milestone): Started from merged PR36/main `f258ce1`.
  Reproduced token7 Red and identified mixed-FMA contraction in emitted ISA/LLVM.
  Added an opaque FP32 register boundary: exact4413 assertions pass, scratch8→0B,
  2000-update chain−16.3% versus baseline and−4.5% versus volatile. Seven short pairs
  and two pairs each at4K/32K give paired medians−0.067%/−0.281%/−0.402%, with all
  IDs/final logits bitwise equal. Decline production adoption without a causal
  explanation; retain benchmark/test improvement, immutable raw data and repro.
  Restored exact production binaries and passed22/22 CTest. Work commit `daa8572`;
  next diagnostic is matched native enqueue and sparse actual-chain timing.
  Delivered ready PR37 and verified mergeability. Auto-subscription is unavailable
  because the token lacks notification scope; no authentication changes were made.
