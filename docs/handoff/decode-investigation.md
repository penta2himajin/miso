# Handoff: decode-investigation

## Snapshot

- Branch: `ai-written/moe-down-fp16-staging`
- Last work commit: ea81b00 @ 2026-09-26 JST
- Base: `d3dd1a6` (main after investigation PR #28)
- Working tree: clean after handoff commit
- Last session: 2026-09-26 JST (Cursor continuing Codex WIP)

## Status

ready-for-review — MoE-down **vector FP16 LDS staging KEEP** (paired +3.40% short decode).

## Next action

Review/merge the FP16-staging PR; next short-context experiment can target dense GEMV
(per investigation plan). Long-context attention work remains a separate track.

## Verification

- Paired short e2e: median **141.1 → 145.9 tok/s** (+3.40%); pp512 ~1440.
- Context probes: 17 **+2.47%**, 4k **+2.39%**, 32k **+1.98%** (no regression).
- `ctest --preset default -j1`: 21/21 pass.
- Resources: LDS 18512→9296; VGPR/occ unchanged; scratch 0.
- Artifacts: `bench/moe/results/2026-09-26-fp16-staging/` (incl. rejected scalar).

## Context pointers

- Investigation: `docs/research/decode-optimization-2026-09-26.md`
- Kernel: `kernels/moe_decode.hpp`, `kernels/gemv_common.hpp`
- Tests: `tests/test_moe_down_staging.hip`, `tests/moe_down_reference.hpp`
- Prior rejects: `docs/handoff/mi50-engine.md` (M8a/M8b)

## Decisions made

- Adoption gate (≥2% short, ≤1% 4k/32k/pp512 regression) from the investigation report.
- **KEEP** vector FP16 staging (4× float4→half4 LDS writes + FP16 packed reads).
- **REJECT** scalar FP16 staging (+0.85% only).
- M8b closed bets remain closed; this change keeps launch geometry and arithmetic order.

## Failed approaches

- Scalar FP16 staging: correct, LDS halved, but short e2e +0.85% < 2% gate.
- M8b L3 drop of `a[9][512]` without FP16 reuse: LDS↓ but Q6 occ↓ — different bet.

## Session log

- 2026-09-26: Investigation PR #28 (measurement only; next = FP16 staging).
- 2026-09-26 (Codex + Cursor): Implemented TDD FP16 staging; scalar REJECT; vector KEEP
  (+3.4% short e2e; gains at 17/4k/32k). Context probes and full CTest completed after
  Codex session paused mid-finalization.
