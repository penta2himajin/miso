# Roadmap

Order agreed on 2026-09-26 (ADR_005): **M4 prefill → M6 quality evaluation → M7 decode optimisation → M5 MTP**. Every phase ends with a PR per sub-milestone (ADR_003 D19). Numbers are measured on this host unless marked *estimate*.

## Where we are (end of M3)

| Metric | Now | llama.cpp (same GGUF, same GPU) | Ceiling |
|---|---|---|---|
| Decode, short ctx | 109–119 tok/s (9.2 ms/token) | 57.2 tok/s | 379 tok/s (bandwidth floor, `docs/research/ornith-q4km-gguf.md`) |
| Prompt ingestion | ~109 tok/s (runs through the decode path) | 921.5 tok/s (pp512) | ~4.3k tok/s at 100% of FP16 dot2 (5.9 GFLOP/token, *estimate*) |
| Model load | 27.6 s | — | ~7 s (20 GB over PCIe 3.0 x4, measured 3.05 GB/s) |

Decode time per token (from the M3b–M3d breakdowns): 30 DeltaNet layers × ~100 µs, 10 attention layers × ~113 µs, 40 MoE blocks × ~103 µs, LM head ~0.58 ms.

## Phase a — M4: prefill kernels (ADR_001 D3 compute family)

Goal: ingest prompts with compute-bound kernels. The decode path is the correctness oracle (ADR_003 D14): a prompt prefilled in chunks must leave the same state and logits as decoding it token by token, within a measured tolerance.

| Step | Content | Exit |
|---|---|---|
| M4a | Batched session API `prefill(tokens[T])` in chunks, with multi-token embedding / RMSNorm; an oracle test harness comparing final hidden state, logits, KV cache and DeltaNet state against the decode path | Harness in place; prefill (initially through per-token kernels) is bit-equal to decode |
| M4b | Q4_K / Q6_K dense GEMM (weights dequantised to FP16 in LDS tiles, `v_dot2_f32_f16` / `v_pk_fma_f16`) for all dense projections | CPU-reference tests; TFLOP/s reported against the 25.4 TFLOP/s dot2 peak at T = 64 / 512 / 2048 |
| M4c | Causal flash attention over a prompt chunk (LDS tiles), batched RoPE and KV write | vs decode path; vs FP64 reference for long T |
| M4d | Chunked gated delta rule (chunk 64) for DeltaNet, plus batched conv | vs decode recurrence, state included. Measured slower than the register recurrence (pp512 387.8 vs 1397.5 tok/s); prefill keeps the single-pass recurrence |
| M4e | MoE prefill: batched router / top-k, tokens grouped by expert, grouped GEMM | Routing identical to decode path; output within tolerance |
| M4f | Integrate; set the decode/prefill dispatch threshold by measurement; CLI uses prefill; TTFT reported | **pp512 > 921.5 tok/s**. Threshold is 8 tokens (run7); pp512 1392 tok/s; CLI reports TTFT |

Order after M4b (user decision, 2026-09-26): **M4e, then M4c, then M4d**. After M4b, pp512 takes 2.55 s. The per-token MoE takes ~2.05 s of that (80%), attention ~0.15 s, the DeltaNet recurrence ~0.17 s and the dense GEMMs ~0.12 s (`bench/model/results/2026-09-26-run3-rocprof-stats.csv`).

## Phase c — M6: quality evaluation (ADR_003 D18, resolves ADR_002 D9)

Goal: decide between the current Q4_K_M and a pure-Q4_K file on measured quality and speed.

| Step | Content | Exit |
|---|---|---|
| M6a | Wait for `Ornith-1.5-35B-BF16.gguf` (verify SHA256 against the HF etag). Build the pure-Q4_K file with `llama-quantize --pure` and an imatrix; record the recipe and hashes | File builds; the engine loads it (Q4_K only) |
| M6b | Fix the evaluation text (English prose, Japanese, code; recorded with hashes) and run `llama-perplexity --kl-divergence` against Q8_0 base logits for Q4_K_M and pure Q4_K | Mean / p99 KL divergence, top-1 agreement, perplexity for both |
| M6c | Engine decode tok/s and pp512 for both files | Numbers recorded |
| M6d | ADR on the weight file | Decision recorded; tests follow the chosen file |

## Phase b — M7: decode optimisation (toward ADR_002 D12 stages 2 and 3)

Goal: stage 2 (≥ 115 tok/s, borderline now), then stage 3 (**≥ 190 tok/s**, 5.3 ms/token). Each step is measured with the D12 protocol before and after.

| Step | Content | Expected effect (*estimate*) |
|---|---|---|
| M7a | MoE: pair gate/up so `SiLU(gate)·up` is written once; size the down grid so the activation reload is not repeated too often | Measured: MoE 103.6/102.2 → 90.3/87.3 µs (Q6_K/Q4_K). A separate top-8 launch cost 25 µs, so top-8 stays in the gate/up kernel. End to end 113.0 tok/s (was ~103) |
| M7b | Fuse launches that share an input: DeltaNet norm + qkv / z / a / b as one GEMV over concatenated rows; attention q / k / v likewise | Measured: z+a+b always fused (Q4_K); qkv fused too when Q4_K (16/30 layers); attention q+k+v fused when all Q4_K (4/10). Decode 114.3 → **123.4 tok/s**; DeltaNet L5 77.9 µs (was ~88 with only z_ab fuse). Mixed-type layers stay split |
| M7c | K = 4096 GEMV (`ssm_out`, `attn_output`: 20 µs for 4.7 MB) and Q6_K qkv efficiency | Measured: shape-tuned R/grid in `gemv()` — K=4096 and large K=2048 use R=4; Q6_K qkv uses R=2 at grid 480. ssm_out 21.9 → 18.5 µs; decode 123.4 → **125.5 tok/s**. Double-buffered Q6_K weight prefetch spilled (112 B scratch) and was reverted |
| M7d | Attention score loop (LDS-bound, ~65 µs extra at 4k context) | Long-context decode |
| M7e | Measure grid-barrier cost and evaluate the ADR_001 D4 megakernel trigger | ADR: megakernel or not |
| M7f | Parallel weight repack at load | Load 27.6 s → < 10 s |

M7a–M7c together are *estimated* at 9.2 − 2.1 − 0.8 − 0.4 ≈ 5.9 ms/token (~170 tok/s); stage 3 then needs M7e or further per-kernel work.

## After that — M5: MTP self-speculative decoding (ADR_002 D8)

Unchanged from ADR_003 D14. The MoE × MTP interaction (union of routed experts, `docs/research/mi50.md` §8.3) makes the gain uncertain, so it is measured against the optimised M7 baseline.
