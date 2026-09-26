# Decode from first principles (2026-09-27)

Base: `6d10fb5` (tree identical to `43e78b1`, after PRs #29/#31/#32).
Hardware: MI50 gfx906, ROCm 6.3.4, Release; default power profile. No concurrent GPU work.

## Current state and cost inventory

The post-KEEP profile in `../2026-09-27-reprofile/` remains the starting evidence:
short MoE 37.3%, dense projections 32.7%, norms 10.3%, LM head 8.6%; at 32k
attention is 25.7%. The fresh initial baseline is 145.9 tok/s (three runs).

`derive_cost.py` parses only the GGUF header and derives current main-40-layer
geometry. It excludes MTP, accounts for 8/256 routed plus one shared expert,
BF16 router storage and quantized row alignment. `derived-cost.json` records
all tensors and the header hash. Active unique weight payload is 1.934 GB;
state plus conv read/write adds 0.132 GB. With KV read/write the totals for
history 0/4096/32768 are 2.065/2.149/2.736 GB. These are source-level payload
counts, not measured HBM transactions. Padding is allocated but not accessed
by slot loads; small activations and partial-output traffic are not included.

At the measured pure-read reference of 797 GB/s, those volumes correspond to
2.591/2.697/3.433 ms. A reference based on peak pure math is also recorded.
Neither calculation establishes that the mixed-instruction decode path is
HBM-bound. Quant unpacking, LDS, reductions, dependencies and issue throughput
are omitted. Likewise the 0.16% positive timestamp gap rules out large host
starvation between measured kernels, but does not quantify fixed overhead
inside the 365 kernel executions. Norm alone consumes 671 us in 81 tiny kernels.

The LM-head already reads 417 MB in 565 us in that profile (~739 GB/s). Its
bytes-preserving bandwidth headroom against the pure-read reference is only
~41 us/token. A high memory-stall counter alone is not a reason to expect a
large improvement from it.

## Experiments

- Attention combine: separate geometry, coefficient preparation and prefetch
  hypotheses. See `../../../attention/results/2026-09-27-combine-tiles/`.
- Mixed Q4/Q6 GEMV grid: interleave original operators in one launch on the
  caller's stream, preserving logical grids and per-row arithmetic. See
  `../../../gemv/results/2026-09-27-mixed-grid/`.
- Native comparisons: baseline, mixed grid only, combine pipeline only, both.
  `run_variants.py` runs three alternating rounds and captures telemetry with
  the existing `run_measure.py`. Warm-up is discarded by each benchmark.

## Reproduce

Build and save one binary/archive per variant before changing the wiring.
The context probe source and measurement harness are archived in
`../2026-09-26-decode-investigation/`. Compile that probe to an object, then
link it with each saved `libmiso_engine.a` and `libmiso_host.a` separately.

```sh
python3 bench/model/results/2026-09-27-first-principles/derive_cost.py MODEL.gguf derived-cost.json
python3 bench/model/results/2026-09-27-first-principles/run_variants.py BINARY_DIR OUTPUT_DIR --runs 3
```

`BINARY_DIR` contains `{baseline,mixed,pipeline,combined}_decode` and selected
`*_context` binaries. Per-run raw logs contain commands, UTC time and exit codes;
matching `.smi.jsonl` logs record clocks, temperatures, power and utilization.

## Adoption and final validation

**KEEP the combined implementation.** Independent microbenchmarks show that
dimension tiling alone does not improve combine; coefficient preparation plus
four-load prefetch does. Three interleaved native e2e rounds give:

- baseline146.0, mixed-only148.8 (+1.92%), pipeline-only146.4 (+0.27%),
  combined148.2 tok/s (+1.51%); all three combined matched rounds improve.
- pp512 baseline1439.9 -> combined1439.7 tok/s (-0.014%, effectively flat).
- restored ctx17: 146.624 ->147.438 (+0.56%); ctx4096:135.226 ->139.467
  (+3.14%); ctx32768:111.838 ->115.647 (+3.41%).

Each restored context result is the median of three 64-token runs in one
process after a discarded warm-up, from the same snapshot each time. Baseline
and candidate use separate processes; unlike short e2e these are one outer
pair per context. Raw logs and telemetry are committed, and predicted IDs
match across each probe's repeats. This is a measured gain on the golden
prompt/workload, not a guaranteed rate for arbitrary prompts.

The source-level payload/state/KV volume divided by native time is an
approximate read-equivalent proxy: ~303 ->307 GB/s for short ctx273..529, and
~306 ->316 GB/s at32k (~38.4 ->39.7% of797 GB/s). It is not actual HBM traffic
or a counter-based bandwidth measurement. Small scratch/output traffic and
cache effects are excluded; state write bandwidth is represented as read
equivalent.

Final timestamp profile: 88320 dispatches over the last256 complete decode
tokens, **345/token** (20 fewer). Mixed pairs take690.9 us/token, versus784.2
in the earlier post-KEEP profile; combine takes61.7 versus83.7 us/token.
The complete raw CSV/stats are gzip-preserved. Profile span is6.461 ms/token
and positive gap0.145%; these instrumented times are not native throughput.
Remaining family shares: MoE37.8%, dense32.4%, norm10.6%, LMhead8.8%.

Validation: full build;21/21 CTest passes (`ctest.txt`); production combine
vs frozen original across516096 fixture comparisons, plus actual split output
comparisons in every existing FP64 boundary case (`test-final-fp64.txt` in the
combine directory); mixed-grid tests172611 assertions on full/odd row counts,
canaries, both argument orders and both stream forms. No tolerances changed.
An independent code review found no correctness issues. Final resource reports
are in the experiment directories; ISA supports the load/dot/prefetch claims.

Next investigation: decompose the81 tiny norm executions with empty-kernel and
dependency controls, then evaluate a concrete producer/consumer design with
explicit residual ownership and synchronization. The new profile still makes
MoE and dense work the largest budgets. Current results do not justify a
blanket HBM-bound or zero-fusion-headroom assumption.

Profiler stdout is preserved in `profile-short.txt.gz`; its readable text copy
has carriage-return progress output and trailing whitespace normalized. Raw
CSV/stats gzip files are unchanged.
