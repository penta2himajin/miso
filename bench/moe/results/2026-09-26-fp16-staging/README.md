# MoE-down FP16 staging experiment

Base: main `d3dd1a6` after investigation PR #28. MI50 32 GB, ROCm 6.3.4,
Release, gfx906. Weight layout, routing, reduction order, launch count and grid
remain unchanged. Only down's activation staging/reading changes.

## TDD and correctness

`red-build.txt`: new Q4/Q6 slot tests fail because the existing Down reader does
not accept FP16 staged inputs. `test-moe.txt` and `vector-test-moe.txt`: after
implementation, all four MoE tests and 46,940 assertions pass. The new tests
compare packed activation bits and rounded offset sums, and compare all 2,048
output rows against the frozen FP32-LDS kernel for both Q4 (layer 5) and Q6
(layer 0). Cases include FP16 ties and their neighbors, subnormals, signed zero,
cancellation, shuffled expert IDs, zero routing weight, and shared-expert gates
at -100, 0 and 100. Existing golden routing/output bounds are unchanged.

## Variants

- `baseline`: main's FP32 LDS activation staging and per-wave conversion.
- `candidate`: scalar FP16 staging, then direct FP16 packed reads. Correct, but
  the paired short e2e median gain is only 0.85% (140.9 -> 142.1 tok/s), below
  the >=2% adoption gate. `candidate.patch` preserves this rejected intermediate.
- `vector`: the same FP16 consumers, with four contiguous FP32 inputs loaded
  and four FP16 values written together during staging. The final source diff
  is preserved in `vector-candidate.patch`.

The production source, tests and trace artifacts describe only this bounded
staging experiment; no precision/weight-format change or grid retuning is used.

## Resources and ISA

Both FP16 variants reduce allocated LDS from 18,512 to 9,296 bytes. Q4/Q6 VGPRs
remain 83/106, scratch and spills remain zero, and resource ceilings remain 3/2
waves per SIMD. No claim of increased resident occupancy is made.

Full down-kernel disassemblies are in `baseline-down-isa.txt`,
`candidate-down-isa.txt` and `vector-down-isa.txt`. They were obtained by
extracting the gfx906 code objects from the actual saved benchmark executables
with `tools/kernel_resources.py` and disassembling with ROCm `llvm-objdump -d
--mcpu=gfx906 --demangle`. The vector staging loop uses `global_load_dwordx4`
and `ds_write_b64`, rather than scalar loads/half writes. Static per-kernel
activation reads (`ds_read_b128`) drop from 24 to 12. These are static instruction
counts, not dynamic counts: the 96 unrolled FP32-to-FP16 conversions become four
inside the vector staging loop. FP32 conversions for the offset sums remain.

## Timing protocol and reproduction

Saved executable variants run on an otherwise idle GPU, with each benchmark's
warm-up discarded. `run_measure.py` from the preceding investigation captures
the command, exit status and rocm-smi clock/temperature/power samples. Telemetry
covers the whole process, including loading, prefill and warm-up.

- Scalar trial: baseline/candidate repeated three times in alternating order.
- Vector trial: vector/baseline repeated three times, reversing the order.
- The existing e2e harness measures ctx 273..529 and pp512 in each process.
- `micro-*.txt` holds three MoE-only runs per variant; microbenchmarks do not
  establish the end-to-end adoption decision.

Example, from repository root (save the baseline binary before building edits):

```bash
cmake --preset default
cmake --build --preset default
./build/tests/test_moe
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py /tmp/micro.txt 3 ./build/bench/bench_moe
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py /tmp/e2e.txt 3 ./build/bench/bench_decode
```

For paired trials, interleave saved baseline and candidate binaries instead of
running three of one before three of the other. Use >=3 runs; add repeats if
ranges overlap substantially or the gain is near 2%.

`lds-counters.txt` filters only the decode down kernels, excluding grouped
prefill kernels. Run separately for baseline and vector with `rocprof
--timestamp on -i lds-counters.txt -o <output.csv> <bench_moe binary>`. Counter
collection changes scheduling; its timestamps are not production timings.

Console log line endings and trailing whitespace are normalized for Git;
measurement values are unchanged. Raw profiler CSVs are retained separately.

## Context probes (restored session state)

Paired baseline vs vector binaries, 64 tokens, ids_match=yes:

| ctx | baseline tok/s | vector tok/s | change |
|---:|---:|---:|---:|
| 17 | 141.126 | 144.618 | **+2.47%** |
| 4096 | 130.600 | 133.721 | **+2.39%** |
| 32768 | 96.836 | 98.753 | **+1.98%** |

No 4k/32k regression (gate was ≤1% regression).

## LDS bank conflicts (diagnostic)

`rocprof` on `moe_down_kernel` only: baseline Q4/Q6 medians 8.1 / 7.4 → vector 4.7 / 1.5.
Counter collection alters scheduling; not used as absolute latency.

## Full tests

`ctest --preset default -j1`: **21/21 pass** (`ctest-full.txt`).

## Verdict

**KEEP vector FP16 staging.** Short e2e paired median **141.1 → 145.9 tok/s (+3.40%)**,
above the ≥2% gate; pp512 unchanged within noise; context 17/4k/32k all improve;
correctness bit-exact vs frozen FP32-LDS path; LDS 18512→9296 B; VGPR/occ ceilings
unchanged (83/3, 106/2); no spills. Scalar candidate remains **REJECT** (+0.85%).
