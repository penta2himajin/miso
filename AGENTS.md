# miso

## Overview

A from-scratch inference engine for **Ornith-1.5-35B-A3B** (`qwen3_5_moe`: 30 Gated DeltaNet + 10 gated full-attention layers, 256-expert top-8 MoE) on a single **AMD Instinct MI50 32 GB (gfx906)**, written in C++20 + HIP on ROCm 6.3.x. Weights come from the Q4_K_M GGUF, repacked at load time. Prefill and decode use separate kernel families.

Hardware facts: @docs/research/mi50.md. Model file facts: @docs/research/ornith-q4km-gguf.md. Settled decisions: `docs/decisions/ADR_*.md`.

## Project Structure

```
src/          # host code: GGUF reader, weight repack, tokenizer, scheduling, CLI
kernels/      # device code: operators as __device__ functions + thin __global__ wrappers
tests/        # CTest / doctest tests and golden-file fixtures
bench/        # microbenchmarks and raw results (bench/<topic>/results/)
tools/        # offline Python scripts (inventory, golden generation, quality evaluation)
third_party/  # vendored test-only code (doctest)
docs/         # research/, decisions/ (ADRs), handoff/
```

## Development Setup

```bash
# Hooks: pre-commit formats staged C++/HIP, pre-push checks formatting.
git config core.hooksPath git-hooks

# clang-format pinned to 18.1.8 (matches ROCm 6.3 clang 18); Ninja (required generator).
python3 -m pip install --user clang-format==18.1.8 ninja==1.11.1.1

# Ad-hoc hipcc outside CMake: this host's clang picks GCC 12 without libstdc++-12-dev.
export HIPFLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11"
```

ROCm 6.3.x is required (ADR_001). CMake auto-detects the GCC install dir (`MISO_GCC_INSTALL_DIR`).

Python tools (fixtures, golden outputs, inventories) run in a git-ignored venv with pinned packages:

```bash
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
```

## Build & Test

```bash
cmake --preset default          # Ninja, Release, gfx906, build/
cmake --build --preset default  # also prints per-kernel VGPR/SGPR/LDS/scratch/occupancy
ctest --preset default          # runs on the MI50
```

Per-kernel resource reports are written to `build/kernel-resources/<target>.txt` (`tools/kernel_resources.py`). The Makefile generator is rejected: CMake 3.22 does not track HIP header dependencies with it.

## Development Principles

- Report performance numbers with the protocol of ADR_002 D12 (clock / temperature / power logged, raw results committed).
- Hot kernels have their emitted ISA inspected; claims about instruction selection cite the ISA.

## Architectural Boundaries

- `kernels/` never includes host-only headers from `src/`. Operators are `__device__` functions; `__global__` kernels are thin wrappers (ADR_001, "D4 → megakernel").
- Weight repacking is a pure, deterministic function of GGUF bytes (ADR_001, "D5 → offline internal format").
- Runtime dependencies are the HIP runtime only. Adding one requires an ADR (ADR_001 D1).

## Prohibitions

1. Do not use MFMA, BF16 dot, packed-FP32 or float global atomic instructions: gfx906 does not have them (docs/research/mi50.md §4).
2. Do not add vendor BLAS (rocBLAS, hipBLASLt) to the production path; rocBLAS is allowed only as a benchmark reference.
3. Do not add CI configuration (ADR_003 D17) without explicit instruction.

## Git Conventions

- Branches: `ai-written/<topic>`, one per workstream or milestone, delivered by pull request (ADR_003 D19). This replaces the `claude/<topic>` example in the common rules. `main` changes only by merging PRs.
- Agent-authored commits carry the trailer `Co-authored-by: Cursor Agent <cursoragent@cursor.com>`.

## Session Handoff

Long-running workstreams use files under `docs/handoff/` for cross-session continuity (ADR_002 D13). See `docs/handoff-protocol.md` for the full protocol.

- One file per workstream: `docs/handoff/<workstream>.md` (template: `docs/handoff/_template.md`)
- Commit and push the handoff update at the end of every session
- On session start, read the relevant handoff file and confirm the **Next action** with the user before executing.

## Internationalisation

If this project ships a Japanese-facing entry point, follow `docs/i18n-policy.md`:

- Translations are suffix files (`README.ja.md` next to `README.md`); no language directories.
- Only `README.md` and the user-facing introduction tier of `docs/` are in scope. Engineering docs and ADRs stay English-only.
- Each translated file carries a `> Source: <name>.md @ <sha>` header. PRs are never blocked on translation parity.

---

<!-- Common rules below this line apply to every project. -->

## Common Development Rules

### TDD (Red → Green → Refactor)

All implementation work proceeds in this cycle:

1. **Red**: write a failing test that captures the intended behaviour.
2. **Green**: write the minimum code that makes the test pass.
3. **Refactor**: tidy up while keeping tests green.

When a test fails, fix the production code — do not delete, skip, or weaken the test.

### Measure, Don't Conjecture

Base decisions on observed data, not assumptions. Before optimising, claiming a bottleneck, or asserting that something is slow or broken, measure it — profile, benchmark, log, or reproduce. When you report a cause, cite the measurement that supports it.

### Git Conventions

- **Conventional Commits**: `feat:` `fix:` `docs:` `refactor:` `test:` `ci:` `chore:`. Project-specific prefixes (e.g. `data:`, `experiments:`) live in the project's `AGENTS.md`.
- **Branch naming**: use a short prefix for the agent or author followed by a topic, e.g. `claude/<topic>`, `codex/<topic>`, or `human/<topic>`.
- **Trailer**: when an AI agent authors the commit, append a trailer crediting the agent. Do not embed model name or session info in the trailer; put those in the commit body if needed.
- **Pre-push hook**: install via `cp git-hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push` (or `git config core.hooksPath git-hooks`). The hook runs format / lint / clippy before every push. Tests are intentionally omitted — TDD keeps them green at commit time.

### Pull Requests

- **Always ready for review.** Open PRs in the "ready" state, never as drafts. Draft PRs do not fire review-requested events and slow the loop.
- **Auto-subscribe after creating a PR.** Immediately after the PR is created, subscribe to its activity without asking the user. Rationale: the user explicitly opted into the "agent opens and watches its own PRs" workflow at the template level, so the per-PR confirmation is noise. Unsubscribe only when the user says to stop, when the PR merges, or when it is closed unmerged.
- **One PR per workstream**, matching the handoff file. Link the handoff file in the PR body.

### Stream Idle Timeout Mitigation

Cloud agent sessions occasionally fail with `Stream idle timeout - partial response received` on long output. To reduce risk:

1. **Stage long writes.** For long documents or source files, write the skeleton (headings, function signatures, trait stubs) first, then fill each section in follow-up edits. Avoid single blocks larger than ~200 lines.
2. **Watch out after large reads.** Reading a big file (e.g. `Cargo.lock`, large generated modules) and then immediately producing long output is a common trigger. Split into separate turns or excerpt only the relevant portion.
3. **Recover carefully.** A timeout can still leave the file write completed. Run `git status` before retrying so the same content is not written twice.

### Common Prohibitions

1. Do not delete, skip, or comment out existing tests.
2. Do not modify CI configuration without explicit instruction.
3. Do not weaken production code merely to make tests pass.
4. Do not commit credentials, API keys, signed URLs, or anything in `.env*`.
