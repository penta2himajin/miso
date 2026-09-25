# Handoff: mi50-engine

<!--
Current-state sections (everything above "Session log") are overwritten each session.
"Session log" is append-only. Protocol: docs/handoff-protocol.md.
-->

## Snapshot

- Branch: `ai-written/m1-golden` (PR open against `main`)
- Last work commit: `4ae952c` @ 2026-09-25
- Working tree: clean after the handoff commit (`.venv/` is git-ignored)
- Last session: 2026-09-25 22:40 JST

## Status

ready-for-review (M1 complete: GGUF reader, CPU dequant, golden harness; `ctest --preset default` 4/4 green)

## Next action

After the M1b PR is merged, branch `ai-written/m2-gemv` from `main` and start M2 (ADR_003 D14): a decode GEMV for Q4_K weights in two variants (A: on-the-fly INT8 activations + `v_dot4_i32_i8`; B: FP16 activations + `v_dot2_f32_f16`), each tested against the CPU dequant reference and benchmarked for bandwidth against 797 GB/s. In parallel (ADR_003 D18), start downloading `Ornith-1.5-35B-Q8_0.gguf` (37.8 GB) and `Ornith-1.5-35B-BF16.gguf` (71.1 GB).

## Verification

- M1b (this branch): `ctest --preset default` passes; `build/tests/test_golden` runs 3 cases, 0 skipped when the model GGUF is present. `tests/golden/README.md` documents regeneration and the 8/8 teacher-forced check.
- M2 exit: A/B decided and recorded by ADR; bandwidth utilisation reported with the ADR_002 D12 protocol.

## Context pointers

- Hardware facts: `docs/research/mi50.md` (§1 summary, §5 measurements, §8 implications)
- Model file facts: `docs/research/ornith-q4km-gguf.md`
- Decisions: `docs/decisions/ADR_001-engine-foundations.md`, `docs/decisions/ADR_002-build-scope-numerics-ops.md`
- Baseline to beat: llama.cpp 57.2 tok/s tg64, 921.5 tok/s pp512 (`/home/penta/llm-mi50/NOTES-ubuntu.md`)
- Input GGUF: `/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf` (SHA256 `42739874…d41f`)
- Local llama.cpp with `gguf-py` and `llama-quantize`: `/home/penta/llm-mi50/src/llama.cpp` (`d81aef1`, build in `build-hip/`)

## Decisions made

See ADR_001 (D1–D6), ADR_002 (D7–D13), ADR_003 (D14–D19). Pending by measurement: activation format A/B (ADR_002 D9, at M2); pure-Q4_K weight file (ADR_002 D9, evaluated per ADR_003 D18 in parallel with M2).

## Failed approaches

- First cache-sweep microbenchmark: grid stride was a multiple of 2^18, so windows ≤256 KiB re-read one address per thread and overstated L1/L2 bandwidth. Fixed by per-block streaming (`bench/mi50/microbench.hip` `k_sweep`).
- `hipcc` without flags fails with `'cmath' file not found`: clang picks GCC 12 but only `libstdc++-11-dev` is installed. Use `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`.

- CMake 3.22 "Unix Makefiles" with HIP sources: the legacy dependency scanner resolved `#include "axpy.hpp"` only relative to `tests/`, so editing a kernel header did not rebuild the test and a deliberately broken kernel still "passed". Ninja (compiler depfiles) is now required and the Makefile generator is rejected in `CMakeLists.txt`.
- Agent shell `cd` persists between commands: a `cd /tmp` in one probe made the next download land in `/tmp/third_party`. Use absolute paths or `( cd … )` subshells.
- Issuing a file edit and the command that uses it in the same parallel tool batch: the command can run before the edit lands (seen with `make_golden.py`). Run dependent steps sequentially.
- A persistent shell running `set -e` exited on the first failing command and killed the agent shell session. Run throwaway tests in a `( … )` subshell instead.

## Open questions for user

- Merge the M1b PR.

## Session log

- 2026-09-25: Measured MI50 (`docs/research/mi50.md`), inventoried the Q4_K_M GGUF, recorded ADR_001 (language, kernels, prefill/decode split, decode execution, weight source, verification) and ADR_002 (build/test, scope, numerics, tokenizer/API, reference weights, benchmarks, repo ops). Found no published all-Q4_K Ornith GGUF; a pure file can be made with `llama-quantize --pure` from the BF16 GGUF (+13% decode floor, lower precision). Created the public GitHub remote and moved handoff to `docs/handoff/`.
- 2026-09-25 (later): Recorded ADR_003 (decode-first milestones M0–M5, source layout, clang-format 18.1.8 via pre-commit + pre-push check, no CI, Q8_0-based KL evaluation, `ai-written/<topic>` branches with PRs). Added the hooks, reformatted `bench/`, filled the project sections of `AGENTS.md`. Opened the PR from `ai-written/adr-003-tooling` (merged as #1).
- 2026-09-25 (M0): Built the scaffold on `ai-written/m0-scaffold`: CMake Ninja preset with GCC auto-detection, vendored doctest 2.4.12, `miso_kernels`/`miso_host` include boundary, post-build kernel resource report from code-object metadata, smoke operator `axpy` in the `__device__` op + thin wrapper pattern. Tests green; a mutation of `axpy_op` is caught (999 failed assertions). Found and fixed stale HIP header dependencies under the Makefile generator. Merged as #2.
- 2026-09-25 (M1a): On `ai-written/m1-gguf`, added the mmap GGUF v3 reader and F32/F16/BF16/Q4_K/Q6_K CPU dequantisation, bit-exact with gguf-py on real model blocks; the C++ tensor table matches gguf-py for all 753 tensors. A Q4_K scale-index mutation is caught. Allowing FMA contraction did not change results, because Q4_K products (11-bit d × 6-bit scale × 4-bit q) are exact in FP32; `-ffp-contract=off` stays as a guard for other formats. Merged as #3.
- 2026-09-25 (M1b): On `ai-written/m1-golden`, built the golden harness (pinned `.venv`: torch 2.14.0+cpu, transformers 5.17.0, gguf 0.19.0). It inverts llama.cpp's converter rewrites (documented in `docs/research/ornith-q4km-gguf.md`), materialises one layer at a time (8.5 GB peak), and records layer 0 (DeltaNet) and layer 3 (full attention) for a 17-token prompt. Teacher-forced llama.cpp continuation through all 40 layers: 8/8, and 0/8 with the V-head reorder disabled. Golden output regenerates byte-identically. C++ dequantised embeddings equal the golden input bit for bit.
