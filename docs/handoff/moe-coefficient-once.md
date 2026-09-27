# Handoff: MoE coefficients once

## Snapshot

- Branch: `ai-written/moe-coefficient-once`
- Base: main `438fb45` (PR #34 merged)
- Last work commit: `0729f95` @ 2026-09-27
- Working tree: clean after the handoff commit
- Last session: 2026-09-27 12:52 JST

## Status

ready-for-review. [PR #35](https://github.com/penta2himajin/miso/pull/35) is open,
ready (not draft) and mergeable. The branch is pushed and attached to this chat.
Automatic subscription was attempted immediately after creation but GitHub
returned `INSUFFICIENT_SCOPES`: the token lacks `notifications`. No auth changes
were made; this does not block code review or merging.

## Next action

Review and merge the coefficient-only-in-block-0 PR, then agree the next measured
decode hypothesis with the user before starting another optimization.

## Verification

- PR is open and ready (not draft), branch pushed, working tree clean.
- `ctest --preset default`: 21/21 pass; archived `ctest.txt` records 87.97 s.
- `test_moe`: 6 cases / 386625 assertions pass, including bitwise frozen controls.
- New C++/HIP sources pass clang-format 18.1.8; analysis scripts pass py_compile.
- Raw data, clock/temperature/power telemetry, binary hashes and ISA are committed
  under `bench/moe/results/2026-09-27-coefficient-once/`.

## Context pointers

- Production: `kernels/moe_decode.hpp`, optional uniform `compute_weights` flag
  and decode `blockIdx.x == 0` call @ `0729f95`.
- Correctness: `tests/test_moe.hip`, frozen top-k and gate/up controls @ `0729f95`.
- Benchmark: `bench/moe/bench_coefficients.hip` and `frozen_gateup.hpp` @ `0729f95`.
- Protocol, findings and reproduction: result-directory `README.md` @ `0729f95`.
- Summaries: `micro-summary.json` and `summary.json`; complete raw measurements retained.
- Preceding workstream: `docs/handoff/decode-norm-latency.md` (PR #34 is now merged).
- Earlier first-principles decode work: `docs/handoff/decode-first-principles.md`.

## Decisions made

- Keep top-8 ID selection in all 120 workgroups. Only workgroup 0 computes and
  exports routing coefficients; prefill uses default full coefficients.
- Preserve reduction/arithmetic order, precision, grid120/block256, row pipeline,
  unconditional block barrier and kernel graph. No launch/global synchronization
  or runtime dependency is added.
- Candidate ISA branches over coefficient work for nonzero groups. Gate/up VGPR63,
  LDS64 B, scratch0 and compiled ceiling4 remain; SGPR42 to40. Prefill SGPR67 to63
  with unchanged VGPR/LDS/scratch/ceiling. Compiled ceilings are not achieved occupancy.
- Keep the change: gate/up paired medians improve 1.295% (layer0) and 1.051%
  (layer5), all 21 pairs faster per layer. Short native median 148.8 to149.8 tok/s
  (+0.672%, seven pairs; paired median +0.535%). This is a local kernel improvement
  with a small measured short gain, not a universal decode-speedup claim.
- Restored-context medians: 17 +0.402% (two pairs, mixed signs), 4k +0.187%
  (three pairs, mixed signs/effectively flat), 32k +1.343% (two positive pairs).
  The third 4k pair was adaptive confirmation; all earlier observations remain.
- Whole-MoE layer5 has a retained +14.639 us candidate outlier; pooled mean and
  per-process medians differ. Setup exports every group's IDs, so it is a control,
  not production setup latency. Larger layer0 whole-MoE gains may include cache
  or compiler scheduling effects. Native DVFS telemetry is not phase-synchronized.

## Failed approaches

- No alternative production design was pursued. The intentional TDD Red was the
  old top-k API rejecting a fourth argument (`red-build.txt`).
- Initial benchmark build hit an ADL ambiguity between frozen and production
  gate/up operators. Explicitly qualifying the frozen call fixed it; incidental
  failed build logs and successful `final-build.txt` are retained.
- Isolated softmax removal does not show a robust 4k end-to-end gain. Do not
  extend sampling indefinitely or present the small positive median as conclusive.

## Session log

2026-09-27: User selected the MoE softmax-coefficient optimization. Started from
merged main `438fb45`, added failing behavioral tests, implemented the uniform
coefficient flag, and verified bitwise equality against frozen old selection and
gate/up for real and synthetic logits. Independent read-only reviews found no
actionable correctness issue. Full CTest passed. Archived ISA confirms nonzero
workgroups skip coefficient math and retain the final barrier. Measured three
processes of seven rotating micro rounds, seven alternating short native pairs,
and restored-context pairs at17/4k/32k with one extra4k pair after mixed signs.
All measurement processes passed and have telemetry. The result supports keeping
the local simplification while reporting native noise, small samples and outliers.
Delivered ready PR #35 and attached it to the chat. GitHub rejected automatic
subscription because the existing token lacks the notifications scope.
