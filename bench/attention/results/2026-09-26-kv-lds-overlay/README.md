# Attention split K/V LDS overlay (long-context)

Base: main `b9defbe` (after MoE-down FP16 staging PR #29). MI50 32 GB, ROCm 6.3.4,
Release, gfx906. Arithmetic order, FP32 scores/online softmax, `attn_n_splits`, and
launch geometry are unchanged. Only decode `attn_split` LDS layout changes: K and V
no longer occupy LDS at the same time.

## Hypothesis

`attn_split` allocated 42,208 B LDS (Q + padded K + dense V), limiting residency to
one workgroup per CU. Score uses K; the value loop uses V. Overlaying those two
buffers cuts peak LDS to 25,824 B so two workgroups can reside (VGPR already allowed
it). Prefill is unchanged. Combine cost is included in every timed comparison.

## Correctness

`test_attn_splitk` extends the FP64 reference to lengths 32767 / 32768 / 32769
(tile boundaries at 32k) in addition to the existing short/mid cases. Relative L2
error stays ≤ 2e-6 (gate 1e-5). Full `ctest --preset default -j1`: 21/21 pass.

## Resources

| variant | VGPR | SGPR | LDS | scratch | spill | occ |
|---------|-----:|-----:|----:|--------:|------:|----:|
| baseline | 95 | 31 | 42208 | 0 | 0 | 1 |
| candidate | 94 | 33 | 25824 | 0 | 0 | 2 |

## Timing protocol

Saved baseline/candidate binaries, interleaved, GPU otherwise idle. `run_measure.py`
records command, exit status, and rocm-smi clock/temperature/power samples.

- Layer-3 `bench_attention` (prep + split + combine + projections): 3 paired runs.
- Context probes (64 decode tokens, `ids_match=yes`): ctx 17 / 4096 / 32768, 3 paired.
- Short e2e `bench_decode` (ctx 273..529 + pp512): 3 paired.

## Results (median of 3 paired runs)

Layer-3 attention decode (µs/token; lower is better):

| ctx | baseline | candidate | change |
|----:|---------:|----------:|-------:|
| 16 | 91.4 | 92.0 | +0.66% |
| 1024 | 100.0 | 101.6 | +1.60% |
| 4096 | 138.6 | 139.8 | +0.87% |
| 16384 | 258.6 | 191.4 | **−26.0%** |
| 32768 | 399.9 | 281.9 | **−29.5%** |

End-to-end context probes (tok/s):

| ctx | baseline | candidate | change |
|----:|---------:|----------:|-------:|
| 17 | 144.640 | 147.046 | +1.66% |
| 4096 | 134.439 | 135.025 | +0.44% |
| 32768 | 98.989 | 113.089 | **+14.24%** |

Short e2e: **141.3 → 144.8 tok/s (+2.48%)**; pp512 unchanged (+0.05%).

## Decision

**KEEP.** Long-context e2e +14.2% at 32k; attention family latency −26%/−29.5% at
16k/32k; no short/4k regression. Occupancy 1 → 2 matches the LDS diagnosis in
`docs/research/decode-optimization-2026-09-26.md` §3.

## Reproduction

```bash
cmake --preset default && cmake --build --preset default
./build/tests/test_attn_splitk
# Save baseline binary, apply candidate.patch (or build this branch), then:
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py \
  /tmp/attn-micro.txt 1 ./build/bench/bench_attention
```
