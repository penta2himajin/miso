# Handoff: deltanet-fp16-prepare

## Snapshot

- Branch: `ai-written/deltanet-fp16-prepare`
- Last work commit: `cee421d` @2026-09-28
- Working tree: clean after this handoff commit; production sources equal main
- Last session: 2026-09-28 00:06 JST

## Status

ready-for-review: measurement milestone complete; current prototype adoption declined.
Latest origin/main remains `af54ec7` (MoE coefficient-once PR35 merged).

## Next action

Review the measurement PR, then discuss a register-only FP32 rounding-boundary
experiment retaining the token7 regression before any new native A/B measurement.

## Verification

- Four `test_deltanet_prepare` cases /3053 assertions pass, no skips.
- `bench_activation_prepare`: 630 rows each for2000/64 updates, all state/output
  bitwise and finite checks pass; warmup500, 3 processes×7 rounds.
- Half-only native `test_deltanet`/`test_model`:2/2 pass.
- Short native7 pairs:513 generated IDs and final full-vocabulary logits bitwise
  match in all pairs. Paired median+0.405%,4/7 faster, throughput medians both148.9tok/s.
- 4K restored-context2 pairs:65 generated IDs and final logits match;
  throughput paired median−1.274%,0/2 faster. 32K: +0.013%,1/2 faster;
  both pairs also match65 IDs and final logits. Restored repeats match warmup bits.
- Restored-main full build andCTest:22/22 pass,102.30s. Format/diff/Python,
  patch applicability, ISA/proof and source/binary/hash checks pass.
- Restored `bench_decode` SHA256 exactly matches original main `afe4f824...`;
  production sources have no diff. Run format/diff/Python checks before committing.

## Context pointers

- Results and limitations: `bench/deltanet/results/2026-09-27-fp16-prepare/README.md`
- Native reproduction: same directory `NATIVE_REPRO.md`, patches, frozen hashes,
  emitted ISA, raw logs/telemetry and actual final-logits dump.
- Benchmark-local operators: `bench/deltanet/activation_prepare.hpp`
- Fairness/reset checks: `bench/deltanet/bench_activation_prepare.hip`
- Counterexample and guarded trajectories: `tests/test_deltanet_prepare.hip`

## Decisions made

- This milestone measures; it does not change the production engine.
- Preparation improves the periodic single-layer chain by12.26% (Half-only,
  2000 updates), but short native gains are mixed and4K regresses in both pairs.
  Do not extrapolate micro percentages or adopt this prototype on that evidence.
- Correct rounding needs a stored FP32 boundary: live cast fails at token7.
  Current Half-only volatile temporary emits private store/load and scratch8B/thread.
  Its causal contribution to native timing has not been isolated.
- Layer4 uses layer0 input fixture with layer4 weights;64/2000 batches are periodic
  cache-hot proxies, not natural autoregressive state. All defined state/output
  bits are checked after each batch.
- No2% threshold was requested. Prior Dual3-pair logs are provisional because
  original patch/binary archives are unavailable; old causal REJECT is withdrawn.
- First new short baseline process overlapped a restored-main CPU build. Excluding
  pair1 leaves+0.169% median and3/6 faster, so the no-consistent-gain conclusion remains.

## Failed approaches

- Live-register half cast: real34-token/grid32 regression fails at step7. Red
  patch/log are archived; volatile FP32 boundary passes all exact checks.
- Dual-only older native study: three pairs have mixed signs and−0.74% paired
  median; missing run archives prevent a strong causal or reproducibility claim.
- Current Half-only scratch-producing prototype: micro improves, but current
  whole-model evidence does not establish a robust improvement across contexts.

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
