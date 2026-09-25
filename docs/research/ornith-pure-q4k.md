# Pure Q4_K Ornith GGUF

Local file for ADR_002 D9. Every quantised tensor is Q4_K. Norms, the router (`ffn_gate_inp`), `ssm_conv1d` and other 1-D tensors stay F32, which is what `llama-quantize --pure` leaves alone.

## Recipe

```
llama-quantize --pure \
  --imatrix Ornith-1.5-35B-A3B-imatrix.gguf \
  Ornith-1.5-35B-BF16.gguf \
  Ornith-1.5-35B-Q4_K.gguf Q4_K
```

- Tool: `/home/penta/llm-mi50/src/llama.cpp` `d81aef19941e145d04f88fb180ea89a67d052ab5`, binary `build-hip/bin/llama-quantize` (8 threads). Quantize time 736 s.
- Input: `Ornith-1.5-35B-BF16.gguf`, SHA256 `044d7f8b55580720dae0d0d8b58f824024901ed5da62f93c011f6bd8dc45a42b` (matches the Hugging Face etag).
- Importance matrix: `bartowski/Ornith-1.5-35B-A3B-GGUF` `Ornith-1.5-35B-A3B-imatrix.gguf`, SHA256 `8d5b16938373b678b68c3081ef3f6985b789df9b25c18af130909658a47c0228`. The imatrix has no rows for `blk.40` (the MTP block). Those tensors were still quantised; the engine does not load `blk.40`.
- Output: `/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K.gguf`
  - SHA256 `7abd5be5f2610baf40c31a2e333a0be86f7b94e423d86f9ef8ded0c4e3014e70`
  - 19121.08 MiB, 4.52 BPW (from 67764.29 MiB, 16.01 BPW)

`tests/test_pure_q4k.hip` checks that every quantised tensor is Q4_K and that `Model::load` accepts the file, including `output.weight` and `ffn_down_exps`, which Q4_K_M keeps at Q6_K.
