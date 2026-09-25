#pragma once

// Gated full-attention token mixer of one layer, decode path (one token per call).

#include <hip/hip_runtime.h>

#include <cstdint>

#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

namespace miso {

struct AttentionLayer {
  QMatrix q, k, v, o;  // attn_q (q + gate), attn_k, attn_v, attn_output
  DeviceBuffer<float> attn_norm, q_norm, k_norm;

  static AttentionLayer load(const gguf::File& f, int layer);
};

// Inverse RoPE frequencies for the rotated dims, computed as transformers does (FP32).
struct RopeConfig {
  DeviceBuffer<float> inv_freq;

  static RopeConfig from(const gguf::File& f);
};

// FP16 KV cache of one layer: [2 KV heads][max_ctx][256] each for K and V.
struct AttentionCache {
  explicit AttentionCache(unsigned max_ctx);
  unsigned max_ctx;
  unsigned len = 0;
  DeviceBuffer<std::uint16_t> k, v;
};

struct AttentionScratch {
  static constexpr unsigned kMaxSplits = 120;  // per KV head: 240 workgroups, ~4 per CU
  DeviceBuffer<float> xn{2048}, qg{8192}, k{512}, v{512}, q{4096}, core{4096};
  DeviceBuffer<float> part_m{2 * kMaxSplits * 8}, part_l{2 * kMaxSplits * 8};
  DeviceBuffer<float> part_o{2 * kMaxSplits * 8 * 256};
};

// y = Attention(RMSNorm(h)) for the token at position cache.len, appended to the cache.
void attention_decode(const AttentionLayer& w, AttentionCache& cache, AttentionScratch& sc,
                      const RopeConfig& rope, const float* h, float* y, float eps,
                      hipStream_t stream);

}  // namespace miso
