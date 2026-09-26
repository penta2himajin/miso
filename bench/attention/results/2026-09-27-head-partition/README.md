# REJECT: attn_split head-group partition (kGroup 8→4)

Base: main `4814fe5` after re-profile
`bench/model/results/2026-09-27-reprofile/`. Hypothesis from peer review:
fewer query heads per split WG cuts LDS ~26→21 KiB (occ 2→3) and may hide
HBM latency at 32k, where `attn_split` is still **22% / 1.91 ms** of kernel time.

## Change (reverted)

- `kAttnHeadGroup = 4` (two WGs per KV head × split); score uses 128 threads,
  value still 256-dim.
- Launch `dim3(n_splits, 4)` instead of `dim3(n_splits, 2)`.
- Resources: LDS 25824→21168, occ 2→3, VGPR 94→74, scratch 0.
- Correctness: `test_attn_splitk` 25/25 including 32k FP64 boundaries.

## Paired results (median of 3)

Layer-3 attention (µs/token; lower better):

| ctx | baseline | candidate | change |
|----:|---------:|----------:|-------:|
| 16 | 91.5 | 91.5 | 0% |
| 4096 | 139.4 | 143.8 | +3.2% |
| 16384 | 191.0 | 276.7 | **+44.9%** |
| 32768 | 281.6 | 430.3 | **+52.8%** |

Context probes (tok/s):

| ctx | baseline | candidate | change |
|----:|---------:|----------:|-------:|
| 17 | 144.901 | 147.068 | +1.5% |
| 4096 | 134.785 | 134.392 | −0.3% |
| 32768 | 112.121 | 96.866 | **−13.6%** |

Short e2e: 145.9 → 144.9 (−0.7%); pp512 flat.

## Decision

**REJECT.** Doubling K/V reads dominates the occupancy win; long-context
attention gets slower, not faster. Gate was ≥5% at 32k — observed −13.6%.
Do not retry head-group partition without a design that keeps single-read K/V
(e.g. different LDS packing that does not multiply traffic).

Production sources restored to main (`kGroup = 8` overlay-only).
