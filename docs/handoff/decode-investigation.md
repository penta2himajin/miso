# Handoff: decode-investigation

## Snapshot

- Branch: `ai-written/decode-investigation`
- Last work commit: `9a31a7b` @ 2026-09-26 JST
- Base: `33915aa`, main after M8b PR #27
- Working tree: clean after handoff commit
- Last session: 2026-09-26 JST

## Status

ready-for-review — measurement and implementation plan complete; production kernels unchanged.

PR: https://github.com/penta2himajin/miso/pull/28 (ready, pushed). Automatic subscription
could not be set: the GitHub token lacks `notifications` scope and the in-app browser
is signed out. Do not claim that notifications or a background monitor are active.

## Next action

Start a bounded MoE-down FP16 LDS-staging experiment: first add failing Q4/Q6 equivalence tests, then keep current launch geometry and arithmetic order while replacing FP32 staging/conversion temporaries; accept only a paired end-to-end gain under the investigation's gates.

## Verification

- Current baseline: median 141.4 tok/s (`bench_decode`, ctx 273..529).
- Restored-state 64-token probes: 140.827 / 131.746 / 98.107 tok/s at ctx 17 / 4096 / 32768.
- `cmake --build --preset default`; `ctest --preset default -j1`: 21/21 pass.
- Context probe checks repeated predictions and session/KV lengths.
- Profile analyzer verifies dispatch counts, excludes prefill, checks exact span accounting;
  deliberately mixed prefill/decode window rejected.
- First implementation gate: paired >=2% short-context gain (about 144.2 tok/s at this baseline),
  <=1% regression at 4k/32k and pp512, unchanged routing/numerical bounds, no spills.
  Marginal results need more than three repeats. See the research report for the long-context gate.

## Context pointers

- Investigation and experiment details: `docs/research/decode-optimization-2026-09-26.md`.
- Full raw traces, telemetry and reproduction: `bench/model/results/2026-09-26-decode-investigation/README.md`.
- MoE-down: `kernels/moe_decode.hpp`, `src/engine/moe.hip`.
- Activation conversion: `kernels/q4k_gemv.hpp`, `kernels/q6k_gemv.hpp`, `kernels/gemv_common.hpp`.
- Existing tests: `tests/test_moe.hip`, `tests/test_attn_splitk.hip`, `tests/test_model.hip`.
- Previous rejected experiments: `docs/handoff/mi50-engine.md` (M8a/M8b).

## Decisions made

- Short decode profile: MoE 39.3%, dense projections 31.6%, norm 10.2%, LM head 8.3%.
- At 32k, attention core/prep becomes 34.0% (3.422 ms/token); prioritize attention first
  if the target workload is long prompts. Split accounts for 2.959 ms, combine 0.408 ms.
- Kernel gaps are only 0.15% short / 0.10% long; retain ADR_006.
- Profiles run faster than unprofiled benchmarks; counter collection changes scheduling.
  Use their breakdowns diagnostically, not as claimed production latency savings.
- Q4/Q6 down have nonzero LDS-bank-conflict counters; FP16 staging must be checked for
  bank conflicts and register pressure as well as LDS bytes. Resource ceilings are not occupancy.
- No proven L2-miss-latency root cause and no claimed optimization win in this batch.
- Golden MoE layers 0/3 cover Q6 down; add Q4 coverage (e.g. layer 5) in the first experiment.

## Failed approaches

- No production optimization was attempted in this investigation; M8b rejections remain closed.
- Initial single-command hipcc link treated `.a` inputs as HIP source. Compile the probe to
  an object, then link the object and archives (documented in the raw-data README).
- `git diff --check` flagged CRLF in derived CSVs and whitespace from profiler console output.
  Normalize textual whitespace; compressed dispatch CSVs retain original bytes.

## Session log

- 2026-09-26: Investigated main after PR #27, rebuilt and reproduced short baseline at
  141.4 tok/s. Measured restored real session state at 17/4k/32k, separated decode from
  prefill in fresh short/32k timestamp profiles, captured two hardware-counter passes,
  and passed all 21 CTests. Saved raw data and probes. Proposed FP16 activation staging
  before revisiting dense GEMV, with a separate attention plan for long contexts; no
  production code changes or claimed speedup.
