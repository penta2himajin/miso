# ADR_002: Build and test, v1 scope, numerics, tokenizer and API, reference weights, benchmarking, repository operations

- Status: Accepted
- Date: 2026-09-25
- Builds on: ADR_001
- Evidence: `docs/research/mi50.md`, `docs/research/ornith-q4km-gguf.md`

## Decisions

### D7. Build and test: CMake + CTest + vendored doctest

The engine builds with CMake using `hipcc` / the HIP language support of ROCm 6.3.x. Tests are CTest targets written with doctest. The doctest header is vendored into the repository, so building and testing need no network access. doctest is a test-only dependency and does not widen the runtime dependency rule of ADR_001 D1.

### D8. v1 scope: text only, single sequence, MTP-ready

- Text generation only. The GGUF contains no vision encoder.
- One sequence at a time (batch 1). No continuous batching.
- MTP self-speculative decoding is a later phase, but the design must admit it from the start. MTP verification processes 2–4 tokens, so the small-token-count kernel path and the KV/DeltaNet state must support committing a variable number of accepted tokens and rolling back rejected ones.

### D9. Numerics: decided by measurement, with fixed defaults where the model dictates

Fixed:

- Residual stream and all accumulation in FP32.
- KV cache in FP16. A full 262,144-token cache is 5 GiB and fits.
- DeltaNet recurrent state in FP32 (`mamba_ssm_dtype = float32` in the model config).

Chosen by measurement in the first decode GEMV kernel:

- **A**: activations quantised on the fly to INT8, with Q4_K/Q6_K weights consumed through `v_dot4_i32_i8` (the llama.cpp MMVQ approach).
- **B**: FP16 activations through `v_dot2_f32_f16` / `v_pk_fma_f16`.

The winner is chosen on measured bandwidth utilisation and on error against the D11 reference. The result is recorded in a follow-up ADR.

Candidate weight file under evaluation: a **pure Q4_K** GGUF (every quantisable tensor in Q4_K; router, `ssm_conv1d`, norms and 1-D tensors left unquantised). No published file has this layout (see §Findings), so it would be produced locally with `llama-quantize --pure`. Adoption requires a measured quality comparison against the current Q4_K_M. Until then ADR_001 D5 stands, and kernel work starts with Q4_K, which both files share.

### D10. Tokenizer and API: own BPE from GGUF metadata, CLI first

The tokenizer is implemented in C++ from `tokenizer.ggml.*` metadata in the GGUF (`model = gpt2`, `pre = qwen35`), including the pre-tokenisation regex. The chat template is applied by engine code. v1 ships a CLI. An HTTP / OpenAI-compatible server is deferred and requires its own ADR, because it adds a dependency.

### D11. Reference weights: dequantised GGUF weights

Golden outputs (ADR_001 D6) are produced by feeding the `transformers` reference the **dequantised** tensors of the GGUF the engine loads (dequantised with `gguf-py`). The tests then measure engine error only. Quantisation quality is evaluated separately against a higher-precision baseline (Q8_0 / BF16 logits or KL divergence).

### D12. Benchmark protocol and staged targets

Protocol for every reported number:

- Idle GPU; warm-up runs discarded; median of repeated runs.
- sclk / mclk / junction temperature / power logged alongside. A sustained run on this host previously reached 103 °C and throttled sclk to 1386–1606 MHz (`docs/research/mi50.md` §9).
- Metrics: decode tok/s at context 0 / 4k / 32k; pp512; time to first token. Also HBM bandwidth utilisation against the 797 GB/s measured read.
- Results are committed as raw files next to the benchmark.

Staged decode targets (Q4_K_M, batch 1, short context), relative to the measured llama.cpp baseline of 57.2 tok/s and the 379 tok/s bandwidth floor:

1. \> 57.2 tok/s (beat llama.cpp)
2. ≥ 115 tok/s (2× llama.cpp)
3. ≥ 190 tok/s (50% of the floor)

### D13. Repository operations: GitHub remote, file-based handoff

- The repository gets an initial commit and a GitHub remote.
- Session handoff is recorded in `docs/handoff/<workstream>.md`, committed and pushed at the end of each session. This replaces the GitHub-issue medium of the template protocol; `docs/handoff-protocol.md` is updated accordingly.

## Findings: is there an all-Q4_K Ornith GGUF?

Checked 2026-09-25 by parsing the GGUF headers (`tools/gguf_header.py` on the first 64 MiB of each file):

| File | Weight types present |
|---|---|
| `ornith-ai/…-GGUF` Q4_K_M (current) | Q4_K, Q6_K (`ffn_down_exps` ×21, `attn_qkv` ×14, `attn_v` ×7, shared down ×21, `output`) |
| `bartowski/…-GGUF` Q4_K_M | Q4_K, Q5_K, Q6_K, Q8_0, Q4_0 |
| `bartowski/…-GGUF` Q4_K_S | Q4_K, Q5_K, Q6_K, Q8_0, Q4_0 |

Standard k-quant mixes deliberately keep sensitive tensors above 4 bits. A pure file can be produced locally:

- Input: `ornith-ai/Ornith-1.5-35B-A3B-GGUF` `Ornith-1.5-35B-BF16.gguf` (71.1 GB; 681 GB disk free).
- Tool: `llama-quantize --pure --imatrix <imatrix> … Q4_K` (the local build at `/home/penta/llm-mi50/src/llama.cpp`, `d81aef1`). It skips `ffn_gate_inp`, `ssm_conv1d`, `*_norm.weight` and 1-D tensors by design.
- Importance matrix: `bartowski/…-GGUF` `Ornith-1.5-35B-A3B-imatrix.gguf`, or one computed locally.

Effect on the decode floor (derived from the Q4_K_M shapes, 797 GB/s):

| Variant | Weights / token | Floor, ctx 0 | Floor, ctx 32k |
|---|---|---|---|
| Q4_K_M (current) | 1,975.7 MB | 379 tok/s | 288 tok/s |
| pure Q4_K | 1,733.7 MB | 429 tok/s (+13%) | 315 tok/s |
| pure Q4_K, LM head kept Q6_K | 1,864.8 MB | 400 tok/s (+6%) | 299 tok/s |

Precision moves the other way: a pure Q4_K file lowers the precision of every tensor that Q4_K_M keeps at Q6_K. The case for it is speed (+13% floor) and a single weight format for all GEMV/GEMM kernels. Quality must be measured before adoption.

## Consequences

- The first kernels target Q4_K (75% of Q4_K_M tensor bytes: 15.23 of 20.21 GiB, and all quantised tensors of a pure file). Q6_K support is scheduled after the D9 weight-file decision.
- The first GEMV kernel is built twice (A and B) and compared before the rest of the decode path is written.
- A quality-evaluation harness (KL divergence against a higher-precision baseline) is needed before D9's candidate can be decided.
