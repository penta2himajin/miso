# Handoff: attn-head-partition

## Snapshot

- Branch: `ai-written/attn-head-partition`
- Last work: REJECT measured; production restored to main overlay
- Base: `4814fe5`
- Last session: 2026-09-27 JST

## Status

in-progress — head-group partition **REJECT** (−13.6% at 32k). Re-profile done.

## Next action

Short-track next bet after gap audit (0.16% → dead) and long head-partition REJECT:
measure FETCH / achieved BW on `moe_gate_up` and top dense Q4 GEMVs from the
fresh short profile, then pick a latency-hiding or fusion hypothesis that is not
a closed M8b layout reopen.

## Verification

- Re-profile: `bench/model/results/2026-09-27-reprofile/`
- Head-partition A/B: `bench/attention/results/2026-09-27-head-partition/` (REJECT)

## Context pointers

- Peers: `bench/model/results/2026-09-27-post-attn-keep/peer-*.txt`
- Short families: MoE 37%, dense 33%, RMSNorm 10%, LM 9%; gap 0.16%
- 32k: attn family 26%, `attn_split` 22% / 1.91 ms after overlay KEEP

## Decisions made

- Re-profile before coding (peer consensus) — done.
- Short launch/gap bets closed (gap 0.16%).
- **REJECT** kGroup 8→4 head partition (KV traffic dominates occ 3).

## Failed approaches

- attn_split head-group partition (kGroup=4, occ 3): correct, LDS↓, but 32k
  e2e −13.6% and layer-3 attn +53% at 32k.

## Session log

- 2026-09-27: Re-profiled short+32k on `4814fe5`; verified head-partition
  hypothesis → REJECT; production reverted to overlay-only.
