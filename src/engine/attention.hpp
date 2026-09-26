#pragma once

// Gated full-attention token mixer of one layer, decode path (one token per call).

#include <hip/hip_runtime.h>

#include <cstdint>

#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

namespace miso {

struct AttentionLayer {
  // attn_q (8192 rows) and attn_k (512) are both Q4_K in every layer, so they are concatenated
  // into `q` (8704 rows) whenever attn_v differs; when all three match, `q` holds all 9216 rows
  // and k.n == v.n == 0. Only if q/k ever differ do they stay separate. The projection buffer
  // layout is the same either way: [q+gate 8192 | k 512 | v 512].
  QMatrix q;
  QMatrix k, v;
  QMatrix o;  // attn_output
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

// Per-token projections: [q+gate 8192 | k 512 | v 512] = 9216 floats.
struct AttentionScratch {
  static constexpr unsigned kMaxSplits = 120;  // per KV head: 240 workgroups, ~4 per CU
  static constexpr unsigned kProj = 9216;
  DeviceBuffer<float> xn{2048}, proj{kProj}, q{4096}, core{4096};
  DeviceBuffer<float> part_m{2 * kMaxSplits * 8}, part_l{2 * kMaxSplits * 8};
  DeviceBuffer<float> part_o{2 * kMaxSplits * 8 * 256};
  float* qg() { return proj.data(); }
  float* k() { return proj.data() + 8192; }
  float* v() { return proj.data() + 8704; }
};

// Chunk buffers of the prefill path, for up to max_tok tokens.
struct AttentionPrefillScratch {
  explicit AttentionPrefillScratch(unsigned max_tok)
      : in(max_tok, 4096),
        xn(max_tok * 2048),
        proj(max_tok * AttentionScratch::kProj),
        q(max_tok * 4096),
        core(max_tok * 4096) {}
  GemmInput in;
  DeviceBuffer<float> xn, proj, q, core;
  float* qg() { return proj.data(); }
  float* k() { return proj.data() + 8192; }
  float* v() { return proj.data() + 8704; }
};

// h += delta (if delta != nullptr), then y = Attention(RMSNorm(h)) for the token at position
// cache.len, which is appended to the cache.
void attention_decode(const AttentionLayer& w, AttentionCache& cache, AttentionScratch& sc,
                      const RopeConfig& rope, float* h, const float* delta, float* y, float eps,
                      hipStream_t stream);

// The same for n consecutive tokens ([n][2048] arrays): projections as prefill GEMMs, then causal
// flash attention over the chunk (each token sees the cache up to itself).
void attention_prefill(const AttentionLayer& w, AttentionCache& cache, AttentionScratch& sc,
                       AttentionPrefillScratch& ps, const RopeConfig& rope, float* h,
                       const float* delta, float* y, unsigned n, float eps, hipStream_t stream);

}  // namespace miso
