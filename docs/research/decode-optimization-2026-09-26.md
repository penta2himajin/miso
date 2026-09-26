# Decode investigation: measurement and next experiments

Date: 2026-09-26. Engine: `33915aa` (main after M8b / PR #27), MI50 32 GB,
ROCm 6.3.4, Release, gfx906. No production kernel or precision change in this batch.

Raw logs, profiler dispatches, resource report and reproduction probes:
[`bench/model/results/2026-09-26-decode-investigation/`](../../bench/model/results/2026-09-26-decode-investigation/).

## Findings

The existing short-context baseline reproduces: **141.4 tok/s** median of
140.0 / 141.4 / 141.6, versus the previous 140.1. This is baseline variation,
not an optimization. Measured pp512 is 1439.9 tok/s median. The benchmark's
measured decode interval is **ctx 273..529**, not ctx 0.

A separate probe prefills real session state using repeated golden tokens,
restores the same KV and DeltaNet state before each repetition, then greedily
decodes 64 tokens. One warm-up is discarded; three measurements follow:

- ctx **17..81**: **140.827 tok/s**, 7.100916 ms/token.
- ctx **4096..4160**: **131.746 tok/s**, 7.590350 ms/token.
- ctx **32768..32832**: **98.107 tok/s**, 10.192968 ms/token.

All four runs at each context reproduce the same 64 predicted IDs; session
and KV lengths agree. These are synthetic repeated-prompt workloads, not a
representative quality evaluation. The short probe uses prefill and a shorter
generation interval, so compare it within this probe rather than directly to
`bench_decode`. A true empty-context measurement is not included.

GPU telemetry is sampled alongside each command. It includes model loading,
prefill and warm-up, not just measured decode. Median active sclk is 1725 MHz;
observed lower-clock samples are retained in the raw logs. Across the unprofiled
runs, peak junction temperature is 77 C and peak power is 249 W. See
`telemetry-summary.json` and each `.smi.jsonl` for mclk and clock ranges.

## Short-context profile, with prefill excluded

`profile.csv.gz` contains the complete timestamp trace. `summarize_profile.py`
selects the last 256 token embeddings and validates 365 dispatches/token,
256 final argmax kernels, absence of prefill kernels, and the duration/span
accounting identity. The selected dispatch indices are 101105..194544.
A mixed prefill/decode window is rejected by the analyzer.

Shares below use the sum of GPU kernel durations in this decode window:

- **MoE: 39.33%**, 2.678 ms/token: gate/up 1.169, Q6 down 0.660,
  Q4 down 0.566, router 0.282.
- **Dense projections: 31.57%**, 2.150 ms/token.
- **Residual RMSNorm: 10.25%**, 0.698 ms/token, 81 calls/token.
- **LM head: 8.29%**, 0.565 ms/token.
- **DeltaNet recurrence: 6.17%**, 0.420 ms/token.
- **Attention core/preparation: 4.07%**, 0.277 ms/token.
- Embedding and argmax: 0.33%.

The profiler run spans 6.820 ms/token, faster than the unprofiled 7.07 ms/token.
Use these shares for prioritization, not as an exact additive decomposition of
the unprofiled time or proof of a prospective speedup. Positive kernel gaps sum
to **0.1503%** of the traced span. Tiny negative overlaps (18,443 ns in total)
are reported separately rather than silently treated as gaps.

This supports keeping ADR_006's no-megakernel decision. It does **not** prove
that intra-kernel prologues, barriers or launch-associated work are free.

Two classification corrections matter: template values `2048u` / `4096u` in
GEMV kernel names are **K**, not N; Q6 R=1 includes both small attention-V
projections and the LM head. The grid separates the head from the six V calls.
The actual 40-layer decode has 20 Q4 and 20 Q6 expert-down calls; inventory
counts that include the MTP layer must not be used as per-token counts.

### Hardware-counter cross-check

Two counter passes select dispatch range `101105:102199` (rocprof's upper bound
is exclusive: 1,094 dispatches, just short of three complete tokens). Per-kernel
medians are in `counter-medians.csv`; `counters.csv` retains the merged raw
passes. Selected observations:

- Gate/up: VALUBusy 29.67%, active-lane VALUUtilization 83.41%, 480 waves/launch.
- Q4/Q6 down: VALUBusy 20.60% / 27.45%, LDSBankConflict 7.36% / 6.77%,
  480 waves/launch. The nonzero LDS counter makes staging access patterns worth
  checking; FP16 staging must not merely trade fewer bytes for worse conflicts.
- K=2048 / K=4096 Q4 projections: VALUBusy 25.06% / 13.83%.
- DeltaNet recurrence: 128 waves/launch; LM head: 7,680 waves/launch and
  VALUBusy 58.46%.

These are rocprof metric definitions from the installed gfx9/gfx906 XML.
VALUUtilization measures active lanes, **not occupancy**; SQ_WAVES counts waves
over a launch, **not simultaneous residents**. Counter collection perturbs
scheduling (the instrumented measured round falls to roughly 122 tok/s), so do
not use those timestamps or percentages as an unprofiled critical-path model.
No supported measurement here isolates L2-miss latency as the proven root cause.
The activation experiments below remain hypotheses with explicit rejection gates.

## Recommended implementation sequence

### 1. MoE-down FP16 activation staging, as a bounded first experiment

`moe_down_op` currently stages `float a[9][512]` in LDS, then each wave loads,
rounds and repacks it into FP16 `SlotAct`s. The total LDS allocation is 18,512 B.
The compiler report gives Q4/Q6 83/106 VGPR and 3/2 waves per SIMD respectively.
These are resource ceilings, not measured dynamic occupancy. A 120-workgroup
grid can impose a further limit; reducing LDS alone does not establish a win.

Keep the current grid, expert assignment, weight layout, reduction order and
launch count initially. Stage activations as FP16, then consume those bits
without converting through temporary FP32 vectors. The LDS payload becomes
9,216 B plus the existing IDs/coefficients (nominal total 9,296 B). Measure the
compiler's actual allocation and transient register pressure. First convert
while staging inside down; only move conversion into the gate/up epilogue if
that isolated experiment shows a useful cost to remove.

This differs from the rejected M8b removal of LDS: retain the staging/reuse,
and target FP32 load/conversion temporaries. That rejection raised Q6 registers
to 134 and reduced its resource ceiling to one wave/SIMD. Also preserve the
three preloaded `SlotAct`s initially: reloading each step and rolling Q4's loop
have already failed (M8a runs 15/16).

**Red:** add comparisons against the existing path for both Q4 and Q6 down,
including a Q4 layer such as 5 (current golden MoE cases 0/3 are Q6), rounding
boundaries, signs, cancellation and shared-expert contribution. Rounded FP16
bits, min/offset sums and their summation order must match. Require routing
IDs/weights unchanged and existing golden/error bounds unchanged. Keep the
residual stream and accumulation FP32; no ADR_004 precision change is intended.

**Green:** implement only the staging/consumer change. Check emitted ISA and
VGPR/LDS/scratch before end-to-end A/B. Reject spills, Q6 register regressions,
or a microbenchmark-only win. A 20% reduction of the current down component
would save about 0.245 ms in the traced workload (about 3.6% overall); this is
a sensitivity calculation, not a forecast.

### 2. Reuse prepared activations in dense GEMV

Only after experiment 1 establishes whether preparation is worth removing,
apply the same idea to the 100 projection GEMVs/token. `load_slot` currently
repeats FP32 loads, FP16 rounding, packing and offset sums across waves.
Explore producer-side packing in existing norm/DeltaNet/attention epilogues,
with a format-specific consumer; do not add an unconditional preparation launch.
Mixed Q4/Q6 projections require their different lane mappings and rounded sums.
Preserve producer arithmetic order and validate bitwise GEMV equivalence before
full-session tests. For the first A/B, hold weight layout, R and grid constant.

This is a new activation-path hypothesis, not a retry of the rejected Q4
weight-layout, header shuffle, Split-K or hand-scheduling experiments. Existing
ISA inspection found the compiler already overlaps weight loads and dots
(`bench/gemv/results/2026-09-26-run30-m8b-l4-isa-excerpt.txt`).

### 3. Long-context attention is a separate priority

At ctx 32768..32832 the last 64-token profile window (indices 108532..131891)
spends **34.00% / 3.422 ms/token** in attention core/preparation, versus
0.277 ms at short context. Other families change comparatively little:
MoE 2.702 ms, projections 2.202 ms, norm 0.726 ms, head 0.565 ms,
DeltaNet 0.425 ms. Profiled median throughput is 99.279 tok/s versus the
unprofiled 98.107. Positive gaps remain only 0.103% of the traced span.
The long-context attention family consists of 2.959 ms split, 0.408 ms combine
and 0.055 ms preparation; optimize split and count combine as part of the cost.

If long prompts are the main workload, do this experiment before dense GEMV.
The current split kernel allocates 42,208 B LDS, limiting it to one resident
workgroup/CU. K and V each occupy about 16 KiB and Q another 8 KiB; the score
loop and value loop have different access patterns. Investigate a smaller
position tile and/or a different division of the eight query heads, together
with split/combine cost. More splits alone have already been tried: include
combine in every comparison. Reducing query-head sharing also increases K/V
reads; explicitly measure that tradeoff instead of assuming more blocks win.

Keep FP32 queries, score accumulation and online softmax. FP16 query/dot2
already violated the desired accuracy margin in M7d. Add 32768 and boundary
cases to the FP64 attention-reference test (currently it stops at 4096), and
check output plus cache continuation. A 30% reduction in the attention family
would save about 1.03 ms in the 32k trace; this is a sensitivity, not a promise.

## Lower priorities and closed approaches

- **DeltaNet spatial partition:** 32 workgroups with 182 VGPR and 2,576 B LDS
  is an obvious underfilled launch on 60 CUs. It is only 6.17% of short decode.
  Splitting V columns can increase block count, but the final per-head RMSNorm
  needs all columns; cost the required reduction/extra launch and conv-state
  ownership before coding. Do not assume the resource ceiling equals occupancy.
- **Norm fusion:** 10.25% is material, but earlier consumer fusion introduced a
  residual race and was ultimately slower; a residual GEMV epilogue also lost.
  Reopen only with a concrete owner/reduction design and a distinct hypothesis.
- **LM head:** 8.29% of short decode, already a large streaming GEMV. Keep it
  separate from the tiny Q6 V projections in statistics. Vocabulary pruning
  would change semantics; pure Q4 or INT8 adoption requires the deferred quality
  work and an ADR, not an incidental speed experiment.
- **Weight traffic/layout:** M8b run31 reports MoE FETCH/ideal 1.006..1.012.
  That rules against substantial transaction amplification in that probe, not
  against memory latency. Low effective GB/s alone does not identify ALU,
  cache-miss latency or residency as the limiting mechanism.
- Do not repeat M8b sequential gate/up SlotW, two-launch Split-K, down LDS removal,
  weight hand-scheduling or transaction-efficiency repacking without new evidence.
  Keep vendor BLAS, unsupported ISA and new runtime dependencies out of production.

## Acceptance protocol and budget

For each experiment, start with a failing equivalence/correctness test, then
implement one change. Run the relevant GPU tests and full-session continuation
checks without loosening tolerances. Inspect emitted ISA and resource reports;
report claimed instruction changes with the actual disassembly. Run baseline /
candidate in alternating order on an otherwise idle GPU, at least three fresh
runs each (increase repeats for marginal results), recording clocks, temperature
and power. Microbenchmarks diagnose; unprofiled end-to-end runs decide.

Use a **relative >=2% short-context gain** over the paired baseline, with no
material (>1%) 4k/32k or pp512 regression. With this session's 141.4 baseline,
that is approximately **144.2 tok/s**, not the old fixed 143 threshold.
A long-context-only change may instead target >=5% at 32k while keeping short,
4k and pp512 within 1%. Borderline changes require more samples; a single
median above a threshold is insufficient if run-to-run ranges overlap heavily.
These are proposed experiment gates, not changes to ADR_002's milestone targets.

The 190 tok/s stage requires 7.072 -> 5.263 ms/token, approximately **1.81 ms /
25.6%** less time. Even halving the entire MoE family would only predict about
176 tok/s under a fixed-share model. Gate/up tuning alone cannot reach 190.
Expect several independent improvements, or later MTP evaluation, rather than
another series of small weight-loop edits. MTP remains a separate roadmap item;
acceptance rate and rollback/verification costs are unmeasured here.

## Validation of this investigation

- Latest main rebuilt with the repository preset; **21/21 CTest tests pass**
  on the MI50, including model, prefill/continuation and CLI golden output.
- Context probe: identical predictions across warm-up plus three restored-state
  repetitions at each of 17, 4096 and 32768; session and KV positions checked.
- Both profile summaries satisfy exact span accounting and decode dispatch-count
  checks; a deliberately mixed prefill/decode window is rejected.
- Diagnostic Python sources compile; the archived HIP probe is clang-formatted.
- This batch measures and proposes experiments. It does not claim a speedup or
  change production kernels, tests, numerical tolerances, dependencies or CI.
