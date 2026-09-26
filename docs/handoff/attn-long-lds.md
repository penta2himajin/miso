# Handoff: attn-long-lds

## Snapshot

- Branch: `ai-written/attn-long-lds`
- Last work commit: 5966095 @ 2026-09-27 JST
- Base: `b9defbe` (main after MoE-down FP16 PR #29)
- Working tree: clean after handoff commit
- Last session: 2026-09-27 JST (Cursor)

## Status

ready-for-review — decode `attn_split` **K/V LDS overlay KEEP** (32k e2e +14.2%).

## Next action

Review/merge the attn-long-lds PR. Next decode experiment: reopen only with a new
measured hypothesis (head tiling / smaller `kAttnSubChunk` remain open; dense GEMV
xn FP16 reuse was REJECT on PR #30).

## Verification

- Layer-3 attn micro at 32k: 399.9 → 281.9 µs/token (−29.5%).
- Context 32768: 98.989 → 113.089 tok/s (+14.24%); 17/4k no regression.
- Short e2e: 141.3 → 144.8 tok/s (+2.48%); pp512 flat.
- Resources: LDS 42208→25824; occ 1→2; scratch 0.
- `ctest --preset default -j1`: 21/21 (incl. FP64 at 32767/32768/32769).
- Artifacts: `bench/attention/results/2026-09-26-kv-lds-overlay/`

## Context pointers

- Plan: `docs/research/decode-optimization-2026-09-26.md` §3
- Kernel: `kernels/attention_decode.hpp` (`attn_split_op` LDS union)
- Tests: `tests/test_attn_splitk.hip`
- Prior short track: PR #30 (GEMV xn FP16 REJECT); PR #29 (MoE-down FP16 KEEP)

## Decisions made

- **KEEP** overlay of padded-K and dense-V LDS in decode `attn_split` (score then
  value over the same shared slot). FP32 Q/scores/softmax unchanged.
- Include combine in every timed comparison; do not retune `attn_n_splits` in this PR.
- Prefill path left alone (still holds K-only staging separately).

## Failed approaches

- None in this session. Not tried yet: smaller `kAttnSubChunk`, fewer query heads per
  workgroup (would increase K/V traffic — measure explicitly if reopened).

## Session log

- 2026-09-27: Implemented K/V LDS overlay after short-context GEMV xn REJECT (PR #30).
  Extended FP64 attn tests to 32k boundaries; paired micro + context + short e2e all
  favour KEEP; LDS 42→26 KiB and occ 1→2.
