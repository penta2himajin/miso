# Handoff: mi50-engine

<!--
Current-state sections (everything above "Session log") are overwritten each session.
"Session log" is append-only. Protocol: docs/handoff-protocol.md.
-->

## Snapshot

- Branch: `ai-written/m8b-gateup-gemv` (ahead of `main`; docs-only M8b batch)
- Last work commit: 56416d7 @ 2026-09-26
- Working tree: clean after handoff commit
- Last session: 2026-09-26 JST

## Status

ready-for-review (M8b: five implement→measure→peer loops; **no KEEP**. Baseline remains median **140.1 tok/s**.)

## Next action

Open/merge the M8b docs PR if desired; next *code* batch should pick a new bottleneck with fresh profile evidence (peers: L2-miss *latency hiding* / wave-overlap — not weight FETCH volume, which is closed at ~1.01× ideal).

## Verification

- Loops 1–5 results under `bench/moe/results/2026-09-26-run27`…`run31` and `bench/gemv/results/2026-09-26-run30-*`.
- e2e baseline: `bench/model/results/2026-09-26-run26-post-m8a-summary.txt` median **140.1**.
- Loop5: `moe_gate_up` FETCH/ideal **1.006×** (`run31-m8b-l5-txn-efficiency.txt`).

## Context pointers

- Hardware facts: `docs/research/mi50.md` (§1 summary, §5 measurements, §8 implications)
- Model file facts: `docs/research/ornith-q4km-gguf.md`
- Decisions: `docs/decisions/ADR_001` … `ADR_005`; plan: `docs/roadmap.md`
- Baseline to beat: llama.cpp 57.2 tok/s tg64, 921.5 tok/s pp512 (`/home/penta/llm-mi50/NOTES-ubuntu.md`)
- Input GGUF: `/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf`

## Decisions made

- KEEP gate for M8b loops: e2e median **≥143** vs run26 **140.1** (ADR_002 D12).
- MoE weight FETCH volume is closed: layout changes justified by “transaction waste” are unsupported (loop5).
- Q4 GEMV weight-path hand-scheduling closed without edit (loop4 ISA invalidate).

## Failed approaches

- M8b L1: sequential `moe_gate_up` SlotW — lost dual-pipeline overlap (`run27`).
- M8b L2: Q4 GEMV Split-K=2 — launch+reduce overhead (`run28`).
- M8b L3: drop `moe_down` shared `a[9][512]` — LDS 18512→80 but Q6 VGPR↑ occ 2→1; e2e~141 (`run29`).
- M8b L4: Q4 GEMV ISA reschedule — abandoned; compiler already overlaps loads/dots (`run30`).
- M8b L5: gate/up layout for txn efficiency — closed; FETCH≈ideal (`run31`).
- Prior M8a rejects remain closed (see earlier session log).

## Session log

<!-- Append-only. One dated paragraph per session. -->

- 2026-09-26 (M3e-2): On `ai-written/m3e-tokenizer`, built the byte-level BPE tokenizer from GGUF metadata with generated Unicode tables (NFC, categories), the chat template and the `miso` CLI (streaming, one step in flight). Found that the HF regex engine matches nothing for `\p{M}` in the qwen35 split pattern; reproducing that brought 250/261 to 261/261 exact matches (`docs/research/ornith-q4km-gguf.md`). The CLI answers a Japanese chat prompt coherently at 119 tok/s and reproduces llama.cpp's continuation. M3 is complete. Merged as #11.
- 2026-09-26 (roadmap): User chose the order a -> c -> b after M3: M4 prefill, then M6 quality evaluation, then M7 decode optimisation, then M5 MTP. Wrote `docs/roadmap.md` with per-step exit criteria and ADR_005.
- 2026-09-26 (M4a): Roadmap merged as #12. On `ai-written/m4a-prefill-harness`, added the layer-major chunked `prefill()` (multi-token embedding and add+RMSNorm, per-token mixer and MoE kernels for now) and the oracle harness `tests/session_compare.hpp` + `tests/test_prefill.hip`. Prefill is bit-equal to decode for chunk sizes 25, 7 and 1, and one decode step after prefill also agrees. A mutation (dropping the MoE delta for token 0) is caught with 0.58-1.16 relative diffs. The CLI now prefills the prompt. Baseline pp512 = 123.3 tok/s (per-token kernels), the M4 starting point against llama.cpp's 921.5.
- 2026-09-26 (M4b): M4a merged as #13. On `ai-written/m4b-gemm`, built the prefill GEMM (`kernels/qgemm.hpp`). Weights are expanded once per workgroup into LDS as exact FP16 integers (Q4_K `q*sc`; Q6_K `q-32`, scaled per 16-group because `|sc|` reaches 128 and 2.3% of `(q-32)*sc` products are not FP16-exact). It computes with `v_dot2_f32_f16` over 64x64 tiles and reaches ~51% of peak at T >= 512. Wired into `deltanet_prefill` / `attention_prefill` for all dense projections. Prefill is as accurate as decode against the golden layers, and the oracle now uses a justified 2e-3 tolerance plus bit-equality across chunk sizes. pp512 went from 123.3 to 200.7 tok/s; the profile shows the per-token MoE is 80% of prefill.
- 2026-09-26 (M4e): M4b merged as #14. On `ai-written/m4e-moe-prefill`, MoE prefill batches the decode router dot product (routing bit-identical), counting-sorts the 9 assignments per token into tasks of 16, and runs grouped gate/up and down GEMMs that stage activations in LDS. The shared expert is expert 256. The DeltaNet prefill recurrence is one launch per layer (head state in registers), bit-identical to per-token steps. pp512 went from 200.7 to 1014.2 tok/s, past llama.cpp's 921.5. The oracle tolerance widened to 1e-2 / 5e-2 because a top-8 near-tie can route differently; this prompt does not.
- 2026-09-26 (M4c): M4e merged as #15. On `ai-written/m4c-flash-attn`, prefill attention is one causal flash kernel per chunk: 4 query tokens share each staged 32-key sub-chunk, online softmax per row, gate applied in the same launch. RoPE and the KV append are the decode prep, batched over the chunk. FP64 error <= 2e-6. pp512 went from 1014.2 to 1397.5 tok/s.
- 2026-09-26 (M4d): M4c merged as #16. On `ai-written/m4d-deltanet`, the chunk-64 gated delta rule plus batched conv matches the decode recurrence (conv bit-exact, output and state within 5e-7). It spills 214 times and, wired into prefill, measured pp512 at 387.8 tok/s, so the shipping path stays on the register recurrence at 1397.5.
- 2026-09-26 (M4f): M4d merged as #17. On `ai-written/m4f-dispatch`, measured decode vs prefill from 1 to 512 tokens. Prefill wins from 8 tokens (75 vs 57 ms). `ingest` uses that split; the CLI reports TTFT. pp512 is 1392 tok/s. M4 is complete.
- 2026-09-26 (M7a): M4f merged as #18. On `ai-written/m7a-moe-decode`, MoE decode writes SiLU(gate)*up once instead of storing both, sizes the gate/up and down grids at 120, and prefetches the weight loads in the router and gate/up row loops. MoE 103.6/102.2 → 85.6/82.5 us (Q6_K/Q4_K down); decode ~103 → 114.3 tok/s. A separate top-k launch cost 25 us and was reverted; router block/grid made no difference; the down prefetch spilled (151 VGPR). Merged as #20.
- 2026-09-26 (M7b): On `ai-written/m7b-fused-gemv`, `attn_gate`, `ssm_alpha` and `ssm_beta` are concatenated at load into one QMatrix (all Q4_K, K = 2048) so one GEMV writes z, a and b; the prefill seq kernel takes a `zab_stride`. DeltaNet layer 0 100 → 88.2 us, decode 114.3 → 119.2 tok/s (past ADR_002 D12 stage 2). `attn_k` is Q4_K but `attn_v` is Q6_K, so those two cannot be concatenated (an attempt threw at load).
- 2026-09-26 (M7a): User paused M6 (PR #19 left as a draft) and asked for speed. On `ai-written/m7a-moe-decode`, fused `SiLU(gate)*up`, cut the down and gate/up grids to 120, and overlapped gate/up weight loads. MoE layers 88.4 / 85.2 us. Decode 114.1 tok/s. Contiguous row walks and a separate top-8 kernel were slower.
- 2026-09-26 (M7b review): Further fuse when types match: DeltaNet qkv+z_ab for the 16 Q4_K-qkv layers (one 12352-row GEMV); attention q+k+v for the 4 all-Q4_K layers (9216 rows). Prefill uses a shared proj buffer with `qkv_stride` / `proj_stride`. Decode 119.2 → **123.4 tok/s**; DeltaNet L5 77.9 µs. Mixed-type layers unchanged. `test_fused_gemv` checks bit-identity vs separate GEMVs. Merged as #21.
- 2026-09-26 (M7c): On `ai-written/m7c-gemv-eff`, `gemv()` picks rows-per-wave and grid by shape (K=4096 and K=2048 with N≥2048 use R=4; Q6_K qkv uses R=2 at grid 480; LM head uses grid 1920). ssm_out 21.9 → 18.5 µs; decode 123.4 → **125.5 tok/s**. A Q6_K next-row weight prefetch spilled and was reverted.
- 2026-09-26 (M7c review): attn_q and attn_k are Q4_K in all 11 attention layers, so the 7 layers whose attn_v is Q6_K now fuse q+k into one 8704-row GEMV (3 → 2 projection launches). Decode 125.5 → **126.0 tok/s**. Tried and rejected: R=5/R=6 and small grids (microbench-optimal but 1.1 tok/s *slower* end to end) and a Q6_K double-buffered weight prefetch (112 B scratch spill). The lesson is recorded under Failed approaches: tune GEMV dispatch on `bench_decode`, not on the isolated sweep.
- 2026-09-26 (M7d): On `ai-written/m7d-attn-score`, staged V alongside K in LDS for split-K decode and retuned `attn_n_splits` (~6 sub-chunks/split, `kMaxSplits=60`) after a max_splits sweep showed combine dominating at 120. Attention L3 173→126 µs at 4k; decode 126.0→**129.1 tok/s**. FP16 Q/fdot2 and a part_o transpose were measured and reverted.
- 2026-09-26 (M7f): On `ai-written/m7f-parallel-repack`, load went 27.6 → **9.7 s**. The cost was repacking into a freshly faulted pageable buffer per tensor (fresh 28.3 s vs reused 13.6 s for the same 21.8 GB), not pinned-vs-pageable (both 3.05 GB/s warm). Added `LoadStage`: 4 pinned slots with event-gated reuse, copies on a side stream, and a 4-chunk repack/copy pipeline. `test_load_stage` proves staged == pageable byte-for-byte; the first chunk loop copied only the last chunk and the test caught it. Decode unchanged at 127.6 tok/s.

- 2026-09-26 (M8a): Started stage-3 decode loop. Cooperative top-k-in-router measured slower and reverted. Fused post-attention RMSNorm into `moe_router` (40 launches/token removed); decode 127.6 → **128.4 tok/s**. Draft PR + peer review next.

- 2026-09-26 (M8a peer loop): astra found residual race; fixed then median −1 tok/s vs main → reverted fusion. Peer consensus: next is top-k algorithm / MoE bandwidth / producer-epilogue norms — not more redundant consumer norms.

- 2026-09-26 (M8a cont.): Replaced O(n²) top-k with 8× masked argmax per astra peer review; decode median 127.7 → **134.2 tok/s**.

- 2026-09-26 (M8a ×3 loops): down-skip / hdr-shfl / float4-norm / grid retune all no e2e win; wave0 top-k +~2 tok/s → median **136.4**. Stopped after 3 loops as requested.
- 2026-09-26 (M8a ×3 after Astra --high): consult ranked (1) expert-major down (2) Q4 r2 SoA (3) residual GEMV epilogue. All measured and rejected (run9–11, run22–24). Baseline unchanged at **136.4 tok/s**. Stopped.

- 2026-09-26 (M8a measure→4 loops): C-excluded campaign (run25) median **136.3**; MoE/down not HBM-bound; LDS caps down occ at 3. Loops 1–3 on moe_down rejected. Loop4 `add_rmsnorm` block 512 kept: median **141.5 tok/s** (`run18`). Stopped after 4.

- 2026-09-26 (M8b ×5 loops): Branch `ai-written/m8b-gateup-gemv` from post-M8a bottleneck (run26 median **140.1**). L1 sequential SlotW REJECT; L2 Split-K REJECT; L3 drop down `a[9][512]` REJECT (LDS↓ Q6 occ↓); Astra-high at L3 entry then L4 ISA schedule ABANDON; L5 rocprof FETCH/ideal ~1.01× CLOSE. No KEEP. Peers: future = L2-miss latency hiding, not weight volume.
