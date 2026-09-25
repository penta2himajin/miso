# Ornith-1.5-35B-A3B Q4_K_M GGUF — contents and decode traffic

Status: recorded 2026-09-25. Produced by `tools/gguf_inventory.py`; raw output in `tools/results/2026-09-25-ornith-q4km-inventory.txt`.

## File

| Item | Value |
|---|---|
| Local path | `/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf` |
| Source | `ornith-ai/Ornith-1.5-35B-A3B-GGUF`, revision `12393612fd4f730ff5aadc23e9b8f9648aa49ceb` |
| SHA256 | `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f` (local `sha256sum` matches the HF LFS etag) |
| Size | 21,713,463,040 B; tensor data 20.21 GiB |
| Format | GGUF v3, `general.architecture = qwen35moe`, `file_type = 15` (Q4_K_M), imatrix-calibrated (3,636 chunks) |
| Blocks | `block_count = 41`: 40 main layers + `blk.40` MTP layer (`nextn_predict_layers = 1`) |
| Vision | not included (text-only GGUF) |

llama.cpp names DeltaNet tensors `ssm_*` (`ssm.state_size = 128`, `ssm.group_count = 16`, `ssm.time_step_rank = 32`, `ssm.inner_size = 4096`).

## Tensor types

| Tensor (per layer unless noted) | Type(s) | Shape (GGUF order) |
|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps` | Q4_K (all 41) | 2048 × 512 × 256 |
| `ffn_down_exps` | Q4_K ×20, **Q6_K ×21** | 512 × 2048 × 256 |
| `ffn_{gate,up}_shexp` / `ffn_down_shexp` | Q4_K / Q4_K ×20 + Q6_K ×21 | 2048 × 512 / 512 × 2048 |
| `ffn_gate_inp` (router), `ffn_gate_inp_shexp` | **F32** | 2048 × 256 / 2048 |
| `attn_qkv` (DeltaNet) | Q4_K ×16, **Q6_K ×14** | 2048 × 8192 |
| `attn_gate` (DeltaNet z), `ssm_out` | Q4_K | 2048 × 4096 / 4096 × 2048 |
| `ssm_alpha`, `ssm_beta` | Q4_K | 2048 × 32 |
| `ssm_conv1d`, `ssm_dt`, `ssm_a`, `ssm_norm` | F32 | 4 × 8192 / 32 / 32 / 128 |
| `attn_q` (+gate), `attn_k`, `attn_output` (full attention) | Q4_K | 2048 × 8192 / 2048 × 512 / 4096 × 2048 |
| `attn_v` | Q4_K ×4, **Q6_K ×7** | 2048 × 512 |
| norms (`attn_norm`, `post_attention_norm`, q/k norms) | F32 | |
| `token_embd` | Q4_K | 2048 × 248,320 |
| `output` (LM head) | **Q6_K** | 2048 × 248,320 |

The engine must therefore implement Q4_K and Q6_K dequantisation for every GEMV/GEMM family, plus F32/BF16 small tensors.

## Decode bytes per token (40 main layers, batch 1)

| Category | MB / token | Share |
|---|---|---|
| Attention + DeltaNet projections + norms | 788.6 | 39.9% |
| Routed experts (8 of 256) | 609.5 | 30.8% |
| LM head (Q6_K) | 417.2 | 21.1% |
| Router (F32) | 84.2 | 4.3% |
| Shared expert | 76.2 | 3.9% |
| **Weights total** | **1,975.7** | |
| DeltaNet state read + write (FP32) | 125.8 | |
| MTP layer (only when drafting) | 40.8 | |

Floor at the measured 797 GB/s HBM read: **379 tok/s** at empty context, 365 at 4k, 288 at 32k, 167 at 128k (FP16 KV, 20 KiB per context token).

Routed experts are less than a third of decode traffic. Dense projections and the LM head dominate, so their GEMV efficiency matters as much as the MoE path.

## Lossless repack opportunities

F32 tensors whose low 16 bits are all zero (exactly representable as BF16):

- `ffn_gate_inp` (41/41), `ffn_gate_inp_shexp` (41/41), `ssm_conv1d` (30/30), `ssm_dt` (30/30), `ssm_norm` (30/30).
- Not exact: `attn_norm`, `post_attention_norm`, q/k norms, `ssm_a`, `output_norm`, MTP norms. These are small; keep F32.

Storing the router as BF16 saves 42 MB/token (~2% of decode traffic) with no numerical change.

## Measured baseline with this file (llama.cpp, same host)

Recorded in `/home/penta/llm-mi50/NOTES-ubuntu.md` (2026-09-25; upstream llama.cpp ROCm 6.3.4 build, f16 KV, `-c 32768 -ub 1024`):

| Test | Result |
|---|---|
| `llama-bench -fa 1` pp512 / tg64 | 921.5 / 57.2 tok/s |
| same at depth 8192: pp / tg | 810.0 / 55.7 tok/s |
| server, no speculation (3 prompts × 400 tok, temp 0.6) | 55.9 tok/s |
| server, built-in MTP n-max 2 (temp 0.6) | 72.4 tok/s (74.3 at temp 0.2) |
| VRAM in use (no speculation / MTP) | 20.8 / 21.7 GiB |

57.2 tok/s is 15% of the 379 tok/s floor.
