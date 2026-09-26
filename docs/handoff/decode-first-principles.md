# Handoff: decode-first-principles

## Snapshot

- Branch: `ai-written/decode-first-principles`
- Base: `6d10fb5` (main after #32)
- PR: https://github.com/penta2himajin/miso/pull/33 (open, ready for review)
- Last work commit: `8867a79` @ 2026-09-27 JST
- Working tree: clean after handoff commit
- Last session: 2026-09-27 JST

## Status

ready-for-review — KEEP mixed Q4/Q6 projection grid plus prepared/pipelined
attention combine. Native short median146.0 ->148.2 tok/s (+1.51%),
4k135.226 ->139.467 (+3.14%), 32k111.838 ->115.647 (+3.41%); pp512 flat.

## Next action

Review/merge this workstream; then measure fixed execution/dependency costs in
the 81 residual RMSNorm calls before choosing a producer/consumer fusion or
kernel change, preserving residual ownership and explicit synchronization.

## Verification

- Full build and21/21 CTest pass; final additional FP64/production combine test
  passes569369 assertions, including frozen bitwise and actual split-output comparisons.
- Mixed GEMV bitwise/canary/stream tests:172611 assertions; model continuation passes.
- Native12-run interleaved comparison plus restored-context probes and telemetry:
  `bench/model/results/2026-09-27-first-principles/`.
- Final short profile validates345 dispatches/token (previous365), mixed pairs
 690.9 us/token versus earlier784.2, combine61.7 versus83.7.
- Combine60-split micro27.25 ->8.14 us; selected ISA confirms four loads before ordered FMAs.
- Final `build/bench/bench_decode` hash matches the measured combined executable:
  `2b562fab4ab452c806a03dbeb17f5509700f7ad701102a3e91d3388afdb101d6`.

## Context pointers

- Kernels: `kernels/mixed_gemv.hpp`, `kernels/attention_decode.hpp`.
- Wiring: `src/engine/weights.hip`, `deltanet.hip`, `attention.hip`.
- Combine experiments: `bench/attention/results/2026-09-27-combine-tiles/`.
- Mixed grid experiments: `bench/gemv/results/2026-09-27-mixed-grid/`.
- Cost inventory/reproduction/remaining profile:
  `bench/model/results/2026-09-27-first-principles/README.md`.
- Earlier work/rejects: `docs/handoff/attn-head-partition.md` and its result links.

## Decisions made

- KEEP the combination for positive native gains at short/4k/32k and flat prefill.
  Short-only mixed is slightly faster (148.8), while the combination benefits longer contexts.
- Math/precision/reduction order and tolerances unchanged. Production caps splits at60;
  prepared combine handles <=64 and retains the original algorithm above64.
- Mixed projections complete in one kernel on the caller's stream. Keep the original
  logical grids; generic shapes use the separate GEMV fallback.
- Cost inventory: current active weight payload1.934 GB, total payload/state/KV
 2.065/2.149/2.736 GB at history0/4k/32k. This is not measured HBM traffic.
- Small positive timestamp gaps do not quantify all fixed costs inside kernel executions.
  Do not generalize the existing megakernel or stream-overlap rejects into a blanket
  rejection of every small fusion. Respect ADR_006's architectural decision.

## Failed approaches

- Combine dimension tiling alone:60 splits27.23 ->27.30 us, no win.
- Prepared coefficients without output prefetch:27.25 ->19.82 us micro; initial
  unpaired short e2e145.9 ->144.4 was inconclusive. Intermediate version not deployed.
- Existing head partition, K4096 R2, dual-stream and M8b rejects remain documented;
  this session used different hypotheses. The old inactive MoE-down-group skip was
  already tried in M8a; do not repeat it without a materially new measured mechanism.

## Session log

- 2026-09-27 (Codex): Confirmed merged #29/#31/#32 and restarted from the current
  source/data costs. Audited bytes/operations rather than assuming HBM-bound decode.
  TDD and ISA checks isolated combine geometry, coefficient preparation and prefetch;
  implemented exact mixed-type projection grids. Twelve native short runs, restored
  17/4k/32k probes, full CTest and final profile support KEEP. Raw traces, thermal/clock/
  power logs, source patch, resource reports and reproducible scripts committed.

- 2026-09-27 (delivery): Opened and attached ready PR #33. Auto-subscription
  failed: GitHub `updateSubscription` requires `notifications`; the token has
  `gist`, `read:org`, `repo`, `workflow`. Implementation/review delivery is complete;
  no authentication scopes were changed.
