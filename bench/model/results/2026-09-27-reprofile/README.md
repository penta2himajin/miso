# Post-KEEP re-profile (main `4814fe5`)

Unprofiled baseline in `../2026-09-27-post-attn-keep/`. This directory holds
rocprof timestamp traces after KEEP #29+#31.

## Short (`bench_decode`, last 256 tokens)

| family | µs/token | share |
|--------|--------:|------:|
| MoE | 2442 | 37.3% |
| Dense projections | 2136 | 32.7% |
| Residual RMSNorm | 671 | 10.3% |
| LM head | 565 | 8.6% |
| DeltaNet | 419 | 6.4% |
| Attention | 285 | 4.4% |

Positive inter-kernel gap: **0.16%** of span → launch/gap bets are dead for short.
Kernel time 6.54 ms/token; span 6.55 ms/token.

## 32k (`context_probe`, last 64 tokens)

| family | µs/token | share |
|--------|--------:|------:|
| MoE | 2518 | 29.0% |
| Attention | 2233 | **25.7%** |
| Dense projections | 2229 | 25.6% |
| Residual RMSNorm | 701 | 8.1% |
| LM head | 564 | 6.5% |
| DeltaNet | 425 | 4.9% |

`attn_split` alone: **1909.8 µs / 22.0%** (pre-#31 family was 34% / 3.42 ms).
Combine 268 µs, prep 54 µs. Gap 0.12%. Profiled median ~115 tok/s (instrumented).

## Decision from peers + this profile

1. Do not pursue short launch/gap / hipGraph (gap << 2%).
2. First kernel hypothesis: **fewer Q-heads per split WG** (kGroup 8→4) to cut LDS
   ~26→21 KiB and raise occ 2→3; measure 32k split+combine (KV traffic doubles).
3. Short track remains MoE gate/up + dense GEMV after the long bet A/B.
