#pragma once

// The full Ornith-1.5-35B-A3B text model on the decode path: embedding, 40 layers (Gated DeltaNet
// or gated full attention, each followed by the sparse MoE block), output norm, LM head, greedy
// argmax. The MTP layer (blk.40) is not loaded (ADR_002 D8: MTP comes later).

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "engine/attention.hpp"
#include "engine/deltanet.hpp"
#include "engine/moe.hpp"
#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

namespace miso {

struct Model {
  struct Layer {
    std::optional<DeltaNetLayer> deltanet;    // linear_attention layers
    std::optional<AttentionLayer> attention;  // full_attention layers
    DeviceBuffer<float> post_attention_norm;
    MoeLayer moe;
  };

  DeviceBuffer<std::uint8_t> embed;  // token_embd, Q4_K r1
  std::vector<Layer> layers;
  DeviceBuffer<float> output_norm;
  QMatrix lm_head;
  RopeConfig rope;
  float eps;
  unsigned vocab;

  static Model load(const gguf::File& f);
};

// Per-sequence state and scratch. Tokens are read from and predictions written to device memory,
// so decode steps can be enqueued without host round trips.
struct Session {
  Session(const Model& m, unsigned max_ctx);

  unsigned max_ctx, pos = 0;
  std::vector<std::optional<DeltaNetState>> deltanet;
  std::vector<std::optional<AttentionCache>> kv;
  DeltaNetScratch dn_scratch;
  AttentionScratch attn_scratch;
  MoeScratch moe_scratch;
  DeviceBuffer<float> h{2048}, mixer_out{2048}, moe_out{2048}, xn{2048}, logits;
  DeviceBuffer<float> argmax_v{kArgmaxParts};
  DeviceBuffer<int> argmax_i{kArgmaxParts};
  DeviceBuffer<int> tokens, pred;  // [max_ctx]: caller-provided inputs, argmax at each position

  // Prefill chunk buffers: [kMaxChunk][2048] residuals, mixer outputs, normed inputs, MoE outputs.
  static constexpr unsigned kMaxChunk = 512;
  DeviceBuffer<float> chunk_h{kMaxChunk * 2048}, chunk_mixer{kMaxChunk * 2048};
  DeviceBuffer<float> chunk_xn{kMaxChunk * 2048}, chunk_moe{kMaxChunk * 2048};
  DeltaNetPrefillScratch dn_prefill{kMaxChunk};
  AttentionPrefillScratch attn_prefill{kMaxChunk};

  static constexpr unsigned kArgmaxParts = 240;
};

// Runs the token at device address `token` at position s.pos and writes the argmax of the
// next-token logits to s.pred[s.pos]; then advances s.pos.
void decode_step(const Model& m, Session& s, const int* token, hipStream_t stream);

// Ingests n tokens (device array) at positions s.pos .. s.pos + n - 1, layer by layer over chunks
// of at most max_chunk (<= Session::kMaxChunk) tokens; writes the argmax after the last token to
// s.pred[s.pos + n - 1] and advances s.pos by n. Leaves the same state as n decode_steps.
void prefill(const Model& m, Session& s, const int* tokens, unsigned n, unsigned max_chunk,
             hipStream_t stream);

}  // namespace miso
