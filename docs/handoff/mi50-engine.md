# Handoff: mi50-engine

<!--
Current-state sections (everything above "Session log") are overwritten each session.
"Session log" is append-only. Protocol: docs/handoff-protocol.md.
-->

## Snapshot

- Branch: `ai-written/m4a-prefill-harness` (PR open against `main`)
- Last work commit: M4a commit on this branch @ 2026-09-26
- Working tree: clean after the handoff commit (`.venv/` is git-ignored)
- Last session: 2026-09-26 00:30 JST
- Background: `Ornith-1.5-35B-BF16.gguf` downloading (36 of 71 GB at 00:25); Q8_0 done and verified. Log: `/home/penta/llm-mi50/logs/dl-miso-quality.log`. When it finishes, check its SHA256 against the etag in `.cache/huggingface/download/*.metadata`.

## Status

ready-for-review (M4a: prefill session API + oracle harness)

## Next action

After the M4a PR is merged, branch `ai-written/m4b-gemm` from `main` and start M4b (`docs/roadmap.md`): Q4_K / Q6_K dense GEMM for prefill (weights dequantised to FP16 in LDS tiles, `v_dot2_f32_f16`), with CPU-reference tests and a TFLOP/s report at T = 64 / 512 / 2048. Then wire it into `prefill()` for the dense projections and compare against the M4a oracle (`tests/test_prefill.hip`) with a tolerance instead of bit equality.

## Verification

- M4a (done): `test_prefill` compares logits, KV cache, conv and recurrent state, position and prediction of `prefill()` against token-by-token decode; bit-equal for chunk sizes 25 (whole prompt), 7 and 1. Baseline pp512 = 123.3 tok/s (`bench/model/results/2026-09-26-run2-prefill-baseline.txt`).
- M4b exit: CPU-reference GEMM tests; TFLOP/s against the 25.4 TFLOP/s dot2 peak.

## Context pointers

- Hardware facts: `docs/research/mi50.md` (§1 summary, §5 measurements, §8 implications)
- Model file facts: `docs/research/ornith-q4km-gguf.md`
- Decisions: `docs/decisions/ADR_001` … `ADR_005`; plan: `docs/roadmap.md`
- Baseline to beat: llama.cpp 57.2 tok/s tg64, 921.5 tok/s pp512 (`/home/penta/llm-mi50/NOTES-ubuntu.md`)
- Input GGUF: `/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf` (SHA256 `42739874…d41f`)
- Local llama.cpp with `gguf-py` and `llama-quantize`: `/home/penta/llm-mi50/src/llama.cpp` (`d81aef1`, build in `build-hip/`)

## Decisions made

See ADR_001 (D1–D6), ADR_002 (D7–D13), ADR_003 (D14–D19), ADR_004 (FP16 decode activations). Pending by measurement: pure-Q4_K weight file (ADR_002 D9, ADR_003 D18; downloads in progress). Measured input for ADR_001 D4: ~7–9 µs fixed cost per GEMV launch.

## Failed approaches

- First cache-sweep microbenchmark: grid stride was a multiple of 2^18, so windows ≤256 KiB re-read one address per thread and overstated L1/L2 bandwidth. Fixed by per-block streaming (`bench/mi50/microbench.hip` `k_sweep`).
- `hipcc` without flags fails with `'cmath' file not found`: clang picks GCC 12 but only `libstdc++-11-dev` is installed. Use `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`.

- CMake 3.22 "Unix Makefiles" with HIP sources: the legacy dependency scanner resolved `#include "axpy.hpp"` only relative to `tests/`, so editing a kernel header did not rebuild the test and a deliberately broken kernel still "passed". Ninja (compiler depfiles) is now required and the Makefile generator is rejected in `CMakeLists.txt`.
- Agent shell `cd` persists between commands: a `cd /tmp` in one probe made the next download land in `/tmp/third_party`. Use absolute paths or `( cd … )` subshells.
- GEMV rows-per-wave to hide per-row latency on small launches: no gain (16.6–17.7 µs for 4.72 MB at R = 1…8). The small-launch cost is a per-launch fixed cost (~7–9 µs), not row serialisation (`bench/gemv/results/2026-09-25-run5-scaling.txt`).
- A data-dependent `break` in the rows loop, and `g < 2 ? h.y : …` member selection, made the compiler move arrays to LDS / scratch (visible in the kernel resource report). Keep indices constant and select computed values.
- `StrReplace` against files that the pre-commit hook has since reformatted fails silently for the non-matching parts. Re-read the file after a commit before editing it.
- M3b error hunt: assumed the 2e-3 DeltaNet error came from the FP16 1024-offset cancellation; switching to subnormal nibbles did not change the failing rows. The real cause was the Q4_K min term using exact activation sums (ADR_004 amendment). Isolate a stage with dumped intermediates vs an FP64 reference before fixing.
- Python-driven multi-line `str.replace` on kernel files silently skips blocks that clang-format reflowed (hit again in `q6k_gemv.hpp`: function renamed but body unchanged). Grep for the old body after editing.
- Attention decode with one workgroup per query head and per-thread K rows: correct but 0.41 us per context token (`bench/attention/results/2026-09-26-run1.txt`). Replaced by split-K.
- Issuing a file edit and the command that uses it in the same parallel tool batch: the command can run before the edit lands (seen with `make_golden.py`). Run dependent steps sequentially.
- A persistent shell running `set -e` exited on the first failing command and killed the agent shell session. Run throwaway tests in a `( … )` subshell instead.

## Open questions for user

- Merge the M4a PR.

## Session log

- 2026-09-25: Measured MI50 (`docs/research/mi50.md`), inventoried the Q4_K_M GGUF, recorded ADR_001 (language, kernels, prefill/decode split, decode execution, weight source, verification) and ADR_002 (build/test, scope, numerics, tokenizer/API, reference weights, benchmarks, repo ops). Found no published all-Q4_K Ornith GGUF; a pure file can be made with `llama-quantize --pure` from the BF16 GGUF (+13% decode floor, lower precision). Created the public GitHub remote and moved handoff to `docs/handoff/`.
- 2026-09-25 (later): Recorded ADR_003 (decode-first milestones M0–M5, source layout, clang-format 18.1.8 via pre-commit + pre-push check, no CI, Q8_0-based KL evaluation, `ai-written/<topic>` branches with PRs). Added the hooks, reformatted `bench/`, filled the project sections of `AGENTS.md`. Opened the PR from `ai-written/adr-003-tooling` (merged as #1).
- 2026-09-25 (M0): Built the scaffold on `ai-written/m0-scaffold`: CMake Ninja preset with GCC auto-detection, vendored doctest 2.4.12, `miso_kernels`/`miso_host` include boundary, post-build kernel resource report from code-object metadata, smoke operator `axpy` in the `__device__` op + thin wrapper pattern. Tests green; a mutation of `axpy_op` is caught (999 failed assertions). Found and fixed stale HIP header dependencies under the Makefile generator. Merged as #2.
- 2026-09-25 (M1a): On `ai-written/m1-gguf`, added the mmap GGUF v3 reader and F32/F16/BF16/Q4_K/Q6_K CPU dequantisation, bit-exact with gguf-py on real model blocks; the C++ tensor table matches gguf-py for all 753 tensors. A Q4_K scale-index mutation is caught. Allowing FMA contraction did not change results, because Q4_K products (11-bit d × 6-bit scale × 4-bit q) are exact in FP32; `-ffp-contract=off` stays as a guard for other formats. Merged as #3.
- 2026-09-25 (M1b): On `ai-written/m1-golden`, built the golden harness (pinned `.venv`: torch 2.14.0+cpu, transformers 5.17.0, gguf 0.19.0). It inverts llama.cpp's converter rewrites (documented in `docs/research/ornith-q4km-gguf.md`), materialises one layer at a time (8.5 GB peak), and records layer 0 (DeltaNet) and layer 3 (full attention) for a 17-token prompt. Teacher-forced llama.cpp continuation through all 40 layers: 8/8, and 0/8 with the V-head reorder disabled. Golden output regenerates byte-identically. C++ dequantised embeddings equal the golden input bit for bit. Merged as #4.
- 2026-09-25 (M2): On `ai-written/m2-gemv`, built the Q4_K decode GEMV in INT8 and FP16 activation variants with rigorous per-row error bounds, the lossless Q4_K r1 repack, DPP reductions and rows-per-wave. v1 FP16 was instruction-bound (63%); v2 reaches 83% (FP16) / 87% (INT8) of 797 GB/s on a 141 MB launch. On decode-sized 4.7 MB launches both take ~16–17 µs because of a ~7–9 µs per-launch fixed cost. ADR_004 selects FP16 (21–37× lower error, 4–12% slower). Started the Q8_0/BF16 downloads for ADR_003 D18. Merged as #5.
- 2026-09-25 (M3a): User approved the M3a–M3e breakdown. On `ai-written/m3a-norm-embed-q6k`: fused add+RMSNorm (golden post-attention norm within bound, residual bit-exact), Q4_K r1 embedding lookup (bit-exact with the golden input), Q6_K r1 row layout (lossless) and Q6_K FP16 GEMV (within 7% of bound). LM head 417 MB at 713 GB/s (89.5%); 13.8 MB DeltaNet qkv at 27.6 us, consistent with the ~8 us per-launch fixed cost. DPP reductions moved to `kernels/wave.hpp`. Merged as #6.
- 2026-09-26 (M3b): On `ai-written/m3b-deltanet`, built the Gated DeltaNet decode step (conv ping-pong state, delta rule in registers, gated norm) and the `src/engine` HIP library (weight upload/repack, GEMV dispatch, DeltaNet layer). The first run gave 2.0e-3 vs golden; stage-by-stage comparison with an FP64 reference traced it to the `ssm_out` GEMV. Fixed the Q4_K min-term rounding inconsistency and replaced the 1024-biased FP16 nibbles with exact subnormals (ADR_004 amendment). The layer now matches the FP16-rounding prediction (2.96e-4); the GEMVs got more accurate and slightly faster. One DeltaNet layer takes ~100 us/token, dominated by the fixed cost of 7 small launches (rocprof breakdown in `bench/deltanet/results/`). Merged as #7.
- 2026-09-26 (M3c): On `ai-written/m3c-attention`, built gated full-attention decode: per-head q/k norm + partial NEOX RoPE + FP16 KV append, then attention. The first version (one workgroup per query head) was correct but scaled at 0.41 us per context token (1.8 ms/layer at 4k). Replaced it with split-K flash decoding (8 query heads per KV-head workgroup, online softmax over 32-position sub-chunks, merge kernel): 177 us at 4k, 334 us at 16k. Golden layer 3 error 4.62e-4, exactly the FP64 prediction for FP16 GEMV inputs + FP16 KV; split-K vs FP64 <= 2e-6 up to 5000 positions. Merged as #8.
- 2026-09-26 (M3d): On `ai-written/m3d-moe`, built the MoE decode path in 3 launches: BF16 router (+ shared gate row), fused gate/up for 8 routed + shared experts with top-8 re-derived per workgroup, fused down + weighting (16 lanes per K = 512 row, Q4_K and Q6_K). Expert ids match golden for all 34 token-layers, output error equals the FP64 prediction (3.6e-4 / 3.9e-4). ~103 us/token per layer; gate/up (47 us) and down (39-44 us) are slowed by per-workgroup redundant top-k and SiLU prologues, candidates for the performance phase. Q8_0 download finished. Merged as #9.
- 2026-09-26 (M3e-1): On `ai-written/m3e-e2e`, assembled the full model (`src/engine/model.*`): device-side tokens and predictions (no host round trip per token), two-stage argmax, residual adds fused into the mixers' input norms. End to end, it reproduces llama.cpp's greedy continuation of the golden prompt 8/8, both teacher-forced and free-running. Decode runs at 108.9 tok/s (9.19 ms/token, ctx ~273-529), 1.9x llama.cpp's 57.2 and past ADR_002 D12 stage 1. Model load takes 27.6 s (single-threaded repack plus PCIe 3.0 x4). Merged as #10.
- 2026-09-26 (M3e-2): On `ai-written/m3e-tokenizer`, built the byte-level BPE tokenizer from GGUF metadata with generated Unicode tables (NFC, categories), the chat template and the `miso` CLI (streaming, one step in flight). Found that the HF regex engine matches nothing for `\p{M}` in the qwen35 split pattern; reproducing that brought 250/261 to 261/261 exact matches (`docs/research/ornith-q4km-gguf.md`). The CLI answers a Japanese chat prompt coherently at 119 tok/s and reproduces llama.cpp's continuation. M3 is complete. Merged as #11.
- 2026-09-26 (roadmap): User chose the order a -> c -> b after M3: M4 prefill, then M6 quality evaluation, then M7 decode optimisation, then M5 MTP. Wrote `docs/roadmap.md` with per-step exit criteria and ADR_005.
- 2026-09-26 (M4a): Roadmap merged as #12. On `ai-written/m4a-prefill-harness`, added the layer-major chunked `prefill()` (multi-token embedding and add+RMSNorm, per-token mixer and MoE kernels for now) and the oracle harness `tests/session_compare.hpp` + `tests/test_prefill.hip`. Prefill is bit-equal to decode for chunk sizes 25, 7 and 1, and one decode step after prefill also agrees. A mutation (dropping the MoE delta for token 0) is caught with 0.58-1.16 relative diffs. The CLI now prefills the prompt. Baseline pp512 = 123.3 tok/s (per-token kernels), the M4 starting point against llama.cpp's 921.5.
