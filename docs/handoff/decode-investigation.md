# Handoff: decode-investigation

## Snapshot

- Branch: `ai-written/gemv-fp16-act-reuse`
- Last work commit: 8ed9dda @ 2026-09-26 JST
- Base: `b9defbe` (main after MoE-down FP16 staging PR #29)
- Working tree: clean after handoff commit
- Last session: 2026-09-26 JST

## Status

ready-for-review — short-context GEMV xn FP16-reuse **REJECT** (−0.27% paired).
MoE-down vector FP16 staging remains KEEP on main.

## Next action

User asked short then long: after merging this REJECT docs PR, start **long-context
attention** (split LDS / head tiling; include combine). Optional later short-context
retry: pack `core`/`o` for K=4096 out GEMVs (not covered here).

## Verification

- Paired e2e median baseline 146.0 vs candidate 145.6 (`bench/gemv/results/2026-09-26-gemv-fp16-act/`).
- Equivalence tests kept green; production wiring reverted.

## Context pointers

- Plan: `docs/research/decode-optimization-2026-09-26.md` §2–3
- Reject write-up: `bench/gemv/results/2026-09-26-gemv-fp16-act/README.md`
- Kernel hooks retained: `Q4kGemvParams::x_h`, `AddRmsNormParams::out_h`, `test_gemv_fp16_act.hip`

## Decisions made

- ≥2% short paired gate; xn-only GEMV act reuse failed it.
- Do not ship rmsnorm→xn_h wiring without a measured win.

## Failed approaches

- Decode xn FP16 pack + K=2048 GEMV consumer: bit-exact, paired −0.27% → REJECT.

## Session log

- 2026-09-26: Investigation + MoE-down FP16 staging KEEP (PR #28–#29).
- 2026-09-26: Short GEMV xn FP16 reuse measured and rejected; production wiring reverted.
