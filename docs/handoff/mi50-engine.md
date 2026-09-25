# Handoff: mi50-engine

<!--
Current-state sections (everything above "Session log") are overwritten each session.
"Session log" is append-only. Protocol: docs/handoff-protocol.md.
-->

## Snapshot

- Branch: `main`
- Last work commit: `7297897` @ 2026-09-25
- Working tree: clean after the handoff commit
- Last session: 2026-09-25 21:45 JST

## Status

in-progress (design discussion; no engine code yet)

## Next action

Settle the remaining discussion items listed under "Open questions for user", then scaffold milestone M0 (CMake + CTest + vendored doctest + HIP smoke test).

## Verification

- Decisions from the discussion are recorded in `docs/decisions/ADR_003-*.md`.
- For M0: `cmake -S . -B build && cmake --build build && ctest --test-dir build` passes on the MI50 host.

## Context pointers

- Hardware facts: `docs/research/mi50.md` (§1 summary, §5 measurements, §8 implications)
- Model file facts: `docs/research/ornith-q4km-gguf.md`
- Decisions: `docs/decisions/ADR_001-engine-foundations.md`, `docs/decisions/ADR_002-build-scope-numerics-ops.md`
- Baseline to beat: llama.cpp 57.2 tok/s tg64, 921.5 tok/s pp512 (`/home/penta/llm-mi50/NOTES-ubuntu.md`)
- Input GGUF: `/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf` (SHA256 `42739874…d41f`)
- Local llama.cpp with `gguf-py` and `llama-quantize`: `/home/penta/llm-mi50/src/llama.cpp` (`d81aef1`, build in `build-hip/`)

## Decisions made

See ADR_001 (D1–D6) and ADR_002 (D7–D13). Pending: pure-Q4_K weight file (ADR_002 D9), activation format A/B (ADR_002 D9).

## Failed approaches

- First cache-sweep microbenchmark: grid stride was a multiple of 2^18, so windows ≤256 KiB re-read one address per thread and overstated L1/L2 bandwidth. Fixed by per-block streaming (`bench/mi50/microbench.hip` `k_sweep`).
- `hipcc` without flags fails with `'cmath' file not found`: clang picks GCC 12 but only `libstdc++-11-dev` is installed. Use `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`.

## Open questions for user

- Milestone order and first TDD target.
- Source layout and module boundaries.
- Formatting / warnings policy and whether the pre-push hook gains C++ checks.
- CI (GitHub runners cannot run gfx906).
- Quality-evaluation baseline for the pure-Q4_K candidate, and whether to start the 71.1 GB BF16 download.
- Branch / PR policy (push to `main` vs. branch + PR per workstream).

## Session log

- 2026-09-25: Measured MI50 (`docs/research/mi50.md`), inventoried the Q4_K_M GGUF, recorded ADR_001 (language, kernels, prefill/decode split, decode execution, weight source, verification) and ADR_002 (build/test, scope, numerics, tokenizer/API, reference weights, benchmarks, repo ops). Found no published all-Q4_K Ornith GGUF; a pure file can be made with `llama-quantize --pure` from the BF16 GGUF (+13% decode floor, lower precision). Created the public GitHub remote and moved handoff to `docs/handoff/`.
