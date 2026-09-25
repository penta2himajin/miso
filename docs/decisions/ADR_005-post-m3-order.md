# ADR_005: Order of work after M3

- Status: Accepted
- Date: 2026-09-26
- Amends: ADR_003 D14 (milestone order after M3)
- Plan: `docs/roadmap.md`

## Context

M3 is complete: the full model decodes at 109–119 tok/s (llama.cpp: 57.2), reproduces llama.cpp's greedy output, and the `miso` CLI produces coherent text. Three workstreams were ready:

- **(a)** prefill kernels (ADR_003 D14 M4). Prompts currently go through the decode path at ~109 tok/s, against llama.cpp's pp512 of 921.5.
- **(c)** the quality evaluation of ADR_003 D18, which decides the ADR_002 D9 pure-Q4_K candidate.
- **(b)** decode optimisation toward ADR_002 D12 stages 2 and 3.

## Decision

Work proceeds **(a) → (c) → (b)**, then MTP:

1. M4 — prefill kernels (unchanged content).
2. M6 — quality evaluation (ADR_003 D18). It runs after M4, so pp512 is measured for both weight files with real prefill kernels.
3. M7 — decode optimisation. It runs after the weight-file decision, so it optimises the kernels of the chosen format.
4. M5 — MTP (ADR_002 D8), measured against the optimised decode baseline.

Milestone numbers keep their identity (M5 stays MTP); new work is M6 and M7.

## Consequences

- The BF16 download continues in the background and is needed only at M6.
- Decode performance work waits until after M6, apart from fixes needed for correctness.
