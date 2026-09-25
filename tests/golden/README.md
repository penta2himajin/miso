# Golden reference outputs

`ornith-layers.gguf` holds FP32 reference tensors for Ornith-1.5-35B-A3B, produced by the Hugging Face `transformers` implementation (`qwen3_5_moe`) running on the **dequantised weights of the Q4_K_M GGUF** the engine loads (ADR_001 D6, ADR_002 D11). The file is GGUF so the C++ tests read it with `src/gguf.hpp`.

## Contents

- Metadata: `golden.prompt`, `golden.token_ids` (17 tokens), `golden.layers` (`[0, 3]`), `golden.model_gguf_sha256`, `golden.hf_config` (HF snapshot of `config.json`), `golden.transformers_version`, `golden.torch_version`.
- `embeddings`: token embeddings of the prompt (input to layer 0).
- Per recorded layer `layer{i}.*` (dims `[2048, 17]` unless noted):
  - `in`, `out`: layer input and output.
  - `mixer_out`: Gated DeltaNet (layer 0) or gated full attention (layer 3) output, before the residual add.
  - `mixer_core` `[4096, 17]`: input of the mixer's output projection (`out_proj` / `o_proj`), in HF head order. Layer 0's V heads are grouped by K head; the GGUF tiles them.
  - `moe_in`: post-attention RMSNorm output; `moe_out`: sparse MoE block output (routed + shared), before the residual add.
  - `router.0` `[256, 17]` router logits; `router.1` / `router.2` `[8, 17]` top-8 routing weights and expert indices, as returned by `Qwen3_5MoeTopKRouter`.

Layer 0 is the first Gated DeltaNet layer. Layer 3 is the first full-attention layer; its input comes from running layers 0–2.

## Regenerate

```bash
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/golden/make_golden.py \
  /home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf \
  --gguf-sha256 42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f
```

About 1 minute on the dev host; peak RSS 8.5 GB (layers are materialised one at a time). Two runs produced byte-identical files (2026-09-25).

## How the weight mapping was validated

`tools/golden/ornith_ref.py` inverts llama.cpp's HF→GGUF rewrites (norm `+1`, `A_log`, conv squeeze, DeltaNet V-head reorder, split experts). To check the inversion end to end, `--verify` runs all 40 layers plus the LM head and teacher-forces llama.cpp's greedy continuation of the prompt (`llama-simple -n 8`, ROCm build `d81aef1`):

| Run | Agreement with llama.cpp's next token |
|---|---|
| Mapping as committed | 8/8 (`東京` `です` `。` ` The` ` capital` ` of` ` Japan` ` is`) |
| V-head reorder disabled (mutation) | 0/8 (garbage tokens) |
