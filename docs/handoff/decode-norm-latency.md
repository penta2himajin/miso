# Handoff: decode-norm-latency

## Snapshot

- Branch: `ai-written/decode-norm-latency`
- Base: `10a00da` (main after PR #33)
- PR: https://github.com/penta2himajin/miso/pull/34 (open, ready for review)
- Last work commit: `d71c45d` @ 2026-09-27 JST
- Working tree: clean after handoff commit
- Last session: 2026-09-27 JST

## Status

ready-for-review — measurement-only Norm/control and trace-perturbation probes.
No production source or dispatch changed. Native representative chain paired
Norm-minus-readwrite0.180/0.330/0.340 us; readwrite-minus-empty5.855/6.040/6.057 us
(GEMV/MoE-Q4/MoE-Q6). Timestamp tracing changes those comparisons substantially.

## Next action

Discuss and select a next experiment, preferably block0-only MoE top-k coefficient
calculation with exact ID/weight tests, setup/full gate-up timing and native decode
comparison; do not prioritize Norm reduction tuning from its traced family total.

## Verification

- New targets `bench_norm`/`bench_norm_chain` build with no production changes.
- Red/Green controls and residual/plain CPU oracles, native/graph true ping-pong
  chains pass; real-weight native/graph four-step outputs/residuals are bitwise equal.
- Independent ring rejects slots1 after a Red/Green check; recorded runs use64.
- Main studies:3 outer processes x7 interleaved rounds; telemetry/raw logs committed.
- Trace study:3 alternating matched native/traced process pairs x3 rounds; raw CSVs
  preserved. Summarizers validate1512 control rows/504 real-chain rows and old
 20736 Norm timestamps over256 tokens.
- Final benchmark device code byte-identical to measured code; production decode
  SHA256 still `2b562fab4ab452c806a03dbeb17f5509700f7ad701102a3e91d3388afdb101d6`.
- Formatting/Python syntax checks pass. Full model CTest not repeated for this
  benchmark-only work; no existing tests or tolerances modified.

## Context pointers

- Design/results/reproduction: `bench/norm/results/2026-09-27-latency/README.md`.
- Full distributions/provenance: same directory `measurements.json.gz`, `manifest.json`.
- Benchmarks: `bench/norm/{bench_norm.hip,bench_model_chain.hip,controls.hpp}`.
- Prior production improvements/rejects: `docs/handoff/decode-first-principles.md`.

## Decisions made

- Use native whole-chain event differences as primary evidence. Differences are
  diagnostic controls, not exact additive costs or legal inference speedup bounds.
- Graph reduces host enqueue work but no material device-time improvement appears
  in these chains; host supply is not the principal explanation in measured cases.
- MoE probes use selected8192-row qkv consumers, omitting production z/a/b combined
  dispatch. Synthetic fixed experts/repeated working sets prohibit81-call extrapolation.
- Keep residual ownership and explicit synchronization in any future fusion.
- Other candidates: K4096 producer-side FP16 preparation (include store/register
  cost), DeltaNet separate partial LDS/remove one overwrite barrier (small budget).
- MoE coefficient redundancy is confirmed in current production ISA: coefficient
  exp0x6b18..0x6e30 precedes first block-zero comparison0x6ec8. Gain is unmeasured.

## Failed approaches

- No production optimization attempted in this measurement milestone.
- Old grid_barrier probe is not a true K-step chain and has duplicate writers;
  its timing is not a Norm dependency calibration.
- An empty timestamped kernel is not a removable-cost estimate: matched GEMV
  empty-minus-direct0.163 us native becomes5.032 us traced.
- Avoid repeating old redundant-consumer Norm, residual-only producer fusion,
  float4 and block128/256 trials without a distinct measured mechanism.

## Session log

- 2026-09-27 (Codex): User approved the Norm cost decomposition after PR #33 merge.
  Created validated controls, real-weight representative chains and telemetry runs;
  inspected ISA and independent reviews. Native/graph/matched tracing show small
  reduction increments and strongly perturbable timestamp accounting. Committed
  raw data/scripts and documented three next hypotheses; inference unchanged.

- 2026-09-27 (delivery): Created and attached ready PR #34. GitHub automatic
  subscription failed because `updateSubscription` requires `notifications` scope;
  the token has `gist`, `read:org`, `repo`, `workflow`. No authentication changes.
