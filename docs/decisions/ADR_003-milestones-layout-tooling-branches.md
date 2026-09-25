# ADR_003: Milestones, source layout, formatting, CI, quality evaluation, branches

- Status: Accepted
- Date: 2026-09-25
- Builds on: ADR_001, ADR_002 (amends D13)

## Decisions

### D14. Milestones: decode first, correctness through the decode path

| Milestone | Content | Exit criterion |
|---|---|---|
| M0 | CMake + CTest + vendored doctest, HIP smoke test, per-kernel VGPR/LDS report | `ctest` green on the MI50 host |
| M1 | GGUF reader, CPU Q4_K dequantisation, golden-output harness (ADR_002 D11) | CPU dequant matches `gguf-py` bit-exactly; golden files for one layer of each type |
| M2 | First decode GEMV (Q4_K), activation format A/B (ADR_002 D9) | A/B decided and recorded by ADR; bandwidth utilisation reported |
| M3 | Full decode path: DeltaNet recurrent, full-attention decode, router + experts, LM head, tokenizer, CLI | Coherent text; layer outputs within tolerance of golden files; > 57.2 tok/s |
| M4 | Prefill kernel families (ADR_001 D3) | pp512 measured; prompt ingestion no longer runs through the decode path |
| M5 | MTP self-speculative decoding (ADR_002 D8) | Accepted-token rate and end-to-end tok/s measured against M3/M4 |

Until M4, prompts are ingested by running the decode kernels one token at a time. That path doubles as the correctness oracle for the prefill kernels.

### D15. Source layout

```
src/        host code: GGUF reader, weight repack, tokenizer, scheduling, CLI
kernels/    device code: operators as __device__ functions + thin __global__ wrappers
tests/      CTest / doctest tests and golden-file fixtures
bench/      microbenchmarks and raw results
tools/      offline scripts (inventory, golden generation, quality evaluation)
third_party/ vendored test-only code (doctest)
docs/       research, decisions, handoff
```

`kernels/` must not include host-only headers from `src/`. This keeps operator bodies callable from a future persistent kernel (ADR_001 "D4 → megakernel").

### D16. Formatting and warnings

- `clang-format` pinned to **18.1.8** (matches ROCm 6.3's clang 18), installed with `python3 -m pip install --user clang-format==18.1.8`. Style: `.clang-format` (Google-based, 100 columns).
- `git-hooks/pre-commit` formats staged `*.h, *.hpp, *.cpp, *.cc, *.hip` files and re-stages them. It refuses files that also have unstaged changes, so unstaged hunks are never committed silently.
- `git-hooks/pre-push` runs `clang-format --dry-run -Werror` over all tracked C++/HIP files.
- Project code compiles with `-Wall -Wextra -Werror`. Vendored `third_party/` code is exempt.

### D17. CI: none for now

GitHub-hosted runners cannot execute gfx906 code. Verification is the pre-commit / pre-push hooks plus `ctest` on the MI50 host before each push. Revisit when a GPU-free test subset becomes large enough to be worth running remotely.

### D18. Quality evaluation for the weight-file decision

- Baseline: `Ornith-1.5-35B-Q8_0.gguf` (37.8 GB) from `ornith-ai/Ornith-1.5-35B-A3B-GGUF`.
- Tool: `llama-perplexity --kl-divergence` from the local llama.cpp build. Candidates are the current Q4_K_M and a locally produced pure Q4_K (ADR_002 D9).
- Downloads (Q8_0, BF16 for `llama-quantize --pure`) and the evaluation run in parallel with M2. The evaluation text and the acceptance threshold are fixed when the harness is written and recorded with the result.

### D19. Branches and pull requests

- All work happens on branches named **`ai-written/<topic>`**, one per workstream or milestone. This replaces the `claude/<topic>` convention of the common rules.
- Each branch is delivered through a pull request, opened ready for review.
- The handoff file (`docs/handoff/<workstream>.md`) is committed and pushed on the working branch at the end of every session. This amends ADR_002 D13.
- `main` changes only by merging pull requests.
