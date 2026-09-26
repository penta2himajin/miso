# Post-KEEP decode baseline + next-hypothesis consultation

Date: 2026-09-27. Engine: main `4814fe5` (after MoE-down FP16 staging PR #29
and attn K/V LDS overlay PR #31). MI50 32 GB, ROCm 6.3.4, Release, gfx906.

Raw post-merge measures: `bench/model/results/2026-09-27-post-attn-keep/`.
Prior investigation (pre-KEEP profile shares):
`docs/research/decode-optimization-2026-09-26.md` and
`bench/model/results/2026-09-26-decode-investigation/`.

## Fresh unprofiled baseline (this session)

Protocol: `run_measure.py`, 3 fresh runs each, GPU otherwise idle, smi logged.

| probe | median | notes |
|-------|-------:|-------|
| short e2e `bench_decode` ctx 273..529 | **146.0 tok/s** | runs 146.8 / 143.9 / 146.0 |
| pp512 | **1439.8 tok/s** | flat vs prior |
| context 17..81 (64 tok) | **144.844 tok/s** | |
| context 4096..4160 | **134.151 tok/s** | |
| context 32768..32832 | **112.204 tok/s** | |

Resources (bench_decode): `attn_split` LDS 25824 occ 2; `moe_down` LDS 9296
(Q4/Q6 occ 3/2). Scratch 0 on these kernels.

## Astra plan §1–3 status (closed)

1. MoE-down vector FP16 LDS staging — **KEEP** (#29): short e2e +3.40% then.
2. Dense GEMV `xn` FP16 act reuse — **REJECT** (#30): paired −0.27%.
3. Long-context attn LDS — **KEEP** (#31): K/V overlay; 32k e2e +14.2% then;
   layer-3 attn −29.5% at 32k. Smaller `kAttnSubChunk` / fewer Q-heads per WG
   were listed in §3 but **not tried** (overlay was the measured LDS bet).

## Still-closed (do not reopen without new evidence)

M8b: sequential SlotW, Split-K, drop down LDS `a[]`, GEMV ISA hand-schedule,
gate/up layout for weight txn efficiency (FETCH≈ideal). M8a: cooperative
top-k-in-router, consumer-side 65× RMSNorm fusion. M7d: FP16 Q / fdot2
(accuracy). More `attn_n_splits` alone (combine already dominates). Vendor
BLAS / MFMA / float atomics / new runtime deps. Scalar FP16 MoE staging
(+0.85% only).

## Profile shares (short decode, pre-KEEP investigation — treat as approximate)

Last 256 tokens, prefill excluded (engine before #29/#31):

- MoE **39.3%** (2.68 ms): gate/up 1.17, Q6 down 0.66, Q4 down 0.57, router 0.28
- Dense projections **31.6%** (2.15 ms)
- Residual RMSNorm **10.3%** (0.70 ms)
- LM head **8.3%** (0.57 ms)
- DeltaNet **6.2%** (0.42 ms)
- Attention **4.1%** (0.28 ms) at short; at 32k was **34%** / 3.42 ms before #31

No fresh rocprof after #29+#31. MoE-down and attn_split costs moved; family
ranking for short is still expected MoE + projections dominant. 32k attention
family should be materially smaller than 3.42 ms after the overlay KEEP.

## Hardware / product constraints

gfx906 MI50: no MFMA, no BF16 dot, no packed-FP32 / float global atomics.
HIP-only runtime. ADR_006: no megakernel. Tune on `bench_decode` / context
probes, not microbench alone. Gate: short ≥2% paired gain, ≤1% regression on
4k/32k/pp512; long-only bets may use ≥5% at 32k with ≤1% short/4k/pp512 hit.
Target context for stage: ~190 tok/s short needs ~1.81 ms/token cut from ~7.07 ms
(investigation framing); current short is **146 tok/s** (~6.85 ms/token).

## Ask

Propose **new** decode (and optionally prefill) hypotheses that are **not** the
closed list above. Prefer bets grounded in the measured shares and resource
reports. For each ranked bet:

1. Concrete mechanism (what changes in which kernel / launch).
2. Why it is not a reopen of a REJECT.
3. What to measure first (red test + which A/B).
4. Rough sensitivity (ms or % if you can defend it from the shares).
5. Risk (correctness, VGPR/LDS/spill, accuracy).

Also: should we re-profile short and/or 32k on `4814fe5` before coding anything?

## Output format (required)

1. **Verdict on next step:** re-profile first / code next / split short vs long tracks.
2. **Ranked next 3 bets** (highest EV first) with mechanism + gate.
3. **Do-not-try** additions if any.
4. **One sentence** on whether further attn tile/head-partition is still worth it
   after the overlay KEEP.

Read the repo if you can: `docs/research/decode-optimization-2026-09-26.md`,
`docs/handoff/attn-long-lds.md`, `docs/research/mi50.md`, `kernels/moe_decode.hpp`,
`kernels/attention_decode.hpp`, `kernels/q4k_gemv.hpp`. Prefer measurement-backed
claims; mark speculation explicitly.
