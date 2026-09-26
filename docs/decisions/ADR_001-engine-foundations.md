# ADR_001: Engine foundations — language, kernels, prefill/decode split, decode execution, weight format, verification

- Status: Accepted
- Date: 2026-09-25
- Scope: inference engine for Ornith-1.5-35B-A3B on a single AMD Instinct MI50 32 GB (gfx906)
- Evidence: `docs/research/mi50.md`, `docs/research/ornith-q4km-gguf.md`

## Context

- gfx906 has no matrix cores, no BF16 arithmetic and no float global atomics, but has full-rate `v_dot2_f32_f16`, `v_dot4_i32_i8` and `v_dot8_i32_i4` (measured 25 / 51 / 102 TOP/s).
- Decode is HBM-bound. Measured streaming read is 797 GB/s. With the Q4_K_M GGUF, the per-token floor is 379 tok/s at empty context; llama.cpp on this host does 57 tok/s (`llama-bench` tg64).
- ROCm support for gfx906 is in maintenance mode; ROCm 6.3.x is the last line that works without patches. hipBLASLt has no gfx906 kernels.
- Empty-kernel launch throughput is 1.25 µs; hipGraph is not faster on ROCm 6.3.4. Cooperative launch is supported; virtual memory management is not.

## Decisions

### D1. Language: C++20 + HIP, single language

Host code and kernels are C++20 compiled by `hipcc` (ROCm 6.3.x). Runtime dependencies are limited to the HIP runtime; any further dependency requires a new ADR.

### D2. Kernels: HIP C++ with AMDGPU builtins

Kernels are written in HIP C++, using `__builtin_amdgcn_*` (dot products, DPP, `ds_bpermute`) where the compiler does not select them. Hot kernels have their ISA inspected. No hand-written assembly and no vendor BLAS in the production path. rocBLAS may be used as a benchmark reference only.

### D3. Prefill and decode use separate kernel families over one shared weight layout

Each operator has a decode kernel (bandwidth-bound: GEMV, recurrent DeltaNet update, flash-decoding) and a prefill kernel (compute-bound: grouped GEMM, chunked delta rule, tiled flash attention). Both read the same device-resident weights. Dispatch between them is by token count, with the threshold set by measurement.

### D4. Decode execution: per-layer fused kernels enqueued ahead on a stream

Decode issues a small number of fused kernels per layer, enqueued on a stream without host synchronisation inside a token.

A persistent megakernel is adopted **only if** measurements taken while pursuing D4 show that it is faster. The trigger is measured inter-kernel overhead (dependent-kernel gaps × kernel count) remaining a material share of token time after per-layer fusion.

### D5. Weight source: the Q4_K_M GGUF, repacked into an internal layout at load time

The input is `Ornith-1.5-35B-Q4_K_M.gguf` from `ornith-ai/Ornith-1.5-35B-A3B-GGUF` (revision `12393612fd4f730ff5aadc23e9b8f9648aa49ceb`, SHA256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`). It mixes Q4_K, Q6_K and F32 tensors. At load, tensors are repacked into engine-specific device layouts.

Switching to an **offline-converted internal format** is adopted **only if** measurements show that load-time repacking is the less efficient option, for example load time dominated by repack cost, or a layout that needs expensive preprocessing.

### D6. Verification: layer-level golden outputs from the `transformers` reference

Golden inputs and outputs per layer (and per operator where useful) are produced by the Hugging Face `transformers` implementation of `qwen3_5_moe`. The C++ tests compare against them with stated tolerances. The reference runs layer-by-layer because host RAM (31 GiB) cannot hold the full model in BF16.

## Keeping the conditional paths open

Both D4 and D5 name a later option that must stay reachable without a rewrite. The following rules are binding from the first kernel onward.

### D4 → megakernel

A megakernel is the limit of fusion: one launch that runs every operator of a token in sequence, with grid-wide synchronisation between dependent steps. Reaching it incrementally requires:

1. **Operators are `__device__` functions; `__global__` kernels are thin wrappers.** An operator takes a work-item range and a parameter struct, so the same body can be called from a per-layer kernel or from a persistent loop.
2. **No operator relies on a kernel boundary for correctness.** Every cross-operator dependency is explicit (a stream-ordered boundary today, a grid barrier or dependency counter later). Intermediates live in preallocated workspace buffers.
3. **One decode block size.** All decode operators are written for a common workgroup size (template parameter), because a megakernel has a single launch shape.
4. **Bounded per-operator resources.** Each decode operator's VGPR and LDS usage is recorded by the build. A megakernel inherits the maximum of its operators, so any operator that inflates it is visible early.
5. **Grid-size independence.** Operators loop over work items (grid-stride) rather than assuming one block per tile, because a cooperative launch is limited to co-resident blocks.

Grid-barrier cost on gfx906 has not been measured yet and is a prerequisite for the D4 trigger.
**Measured 2026-09-26** (`bench/mi50/results/2026-09-26-m7e-grid-barrier.txt`): a grid barrier
costs 1.81 µs at 60×256 against 1.69 µs for a kernel boundary. With dependent-kernel gaps at
0.18% of decode time after per-layer fusion, the trigger is measured false and decode stays
kernel-per-operator — see ADR_006.

### D5 → offline internal format

1. **Repacking is a pure function** from GGUF tensor bytes plus metadata to internal-layout bytes, independent of GPU state and deterministic.
2. **The internal layout is versioned.** An offline file is then just the serialised output of the same function, with a header carrying the layout version and the source GGUF SHA256.
3. **Both load paths produce byte-identical device buffers.** This is enforced by a test once the offline path exists.

Limit: conversion from this GGUF can only be lossless re-arrangement. Examples are interleaving, pre-expanding the packed 6-bit Q4_K scales, and storing BF16-exact F32 tensors as BF16. The router, `ssm_conv1d`, `ssm_dt` and `ssm_norm` tensors were verified to be BF16-exact. Moving to a different quantisation scheme (for example symmetric INT4 for `v_dot8_i32_i4`) requires the BF16 checkpoint and a separate ADR.

## Consequences

- Kernels must implement Q4_K and Q6_K semantics exactly (plus F32 / BF16 small tensors).
- The build must report per-kernel VGPR / LDS usage.
- Pinning to ROCm 6.3.x is a hard constraint until an ADR revisits it.
- Reference-output generation needs the BF16 safetensors checkpoint (~70 GB download; 681 GB free) and a layer-streaming harness.
