# Handoff: attn-head-partition (short-track continues on same PR)

## Snapshot

- Branch: `ai-written/attn-head-partition`
- Base: `4814fe5`
- Last session: 2026-09-27 JST

## Status

in-progress — measurement PR: head-partition REJECT; K=4096 R=2 REJECT;
dual-stream GEMV overlap REJECT (correctness). Production unchanged.

## Next action

Pick another short-track bet that is not: launch/gap, head-partition, K=4096
R=2, dual-stream overlap, or closed M8b layout/FETCH. Candidates after
counters: LM-head MemStall (needs ADR/precision), gate_up LDS-wait (grid
already swept), or a fresh measured hypothesis.

## Verification

- Re-profile + counters: `bench/model/results/2026-09-27-reprofile/`
- Head partition: `bench/attention/results/2026-09-27-head-partition/` REJECT
- K=4096 R=2: `bench/gemv/results/2026-09-27-k4096-r2/` REJECT (e2e +0.0%)
- Dual-stream: `bench/gemv/results/2026-09-27-dual-stream/` REJECT (test_model race)

## Decisions made

- Short gap 0.16% → no launch/gap bets.
- **REJECT** attn head-group partition (−13.6% at 32k).
- **REJECT** Q4 K=4096 R=4→2 (occ 3, e2e flat).
- **REJECT** dual-stream independent GEMV overlap (nondeterministic wrong ids).

## Failed approaches

- attn kGroup 8→4; K=4096 R=2; dual-stream gemv_pair (see result READMEs).

## Session log

- 2026-09-27: Re-profile + counters; three short/long hypotheses measured/rejected
  on this branch; production restored to main kernels/dispatch.
