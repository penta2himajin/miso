#pragma once

// Gated DeltaNet token mixer of one layer, decode path (one token per call).

#include <hip/hip_runtime.h>

#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

namespace miso {

struct DeltaNetLayer {
  // When attn_qkv is Q4_K (same as gate/alpha/beta), qkv holds all 12352 rows and z_ab.n == 0.
  // Otherwise qkv is 8192 rows (often Q6_K) and z_ab is the 4160-row gate+alpha+beta concat.
  QMatrix qkv;
  QMatrix z_ab;
  QMatrix out;  // ssm_out
  DeviceBuffer<float> attn_norm, conv_w, ssm_a, dt_bias, ssm_norm;

  static DeltaNetLayer load(const gguf::File& f, int layer);
};

// Recurrent state carried across tokens: double-buffered conv history and the FP32 delta-rule
// state. Zero at construction (start of a sequence).
struct DeltaNetState {
  DeviceBuffer<float> conv[2]{DeviceBuffer<float>(8192 * 3), DeviceBuffer<float>(8192 * 3)};
  DeviceBuffer<float> s{32 * 128 * 128};
  int parity = 0;

  DeltaNetState() { reset(); }
  void reset();
};

// Per-token projection buffer: [qkv 8192 | z 4096 | a 32 | b 32] = 12352 floats.
struct DeltaNetScratch {
  static constexpr unsigned kProj = 12352;
  DeviceBuffer<float> xn{2048}, proj{kProj}, o{4096};
  float* qkv() { return proj.data(); }
  float* z() { return proj.data() + 8192; }
  float* a() { return proj.data() + 12288; }
  float* b() { return proj.data() + 12320; }
};

// Chunk buffers of the prefill path, for up to max_tok tokens. Same proj layout per token.
struct DeltaNetPrefillScratch {
  explicit DeltaNetPrefillScratch(unsigned max_tok)
      : in(max_tok, 4096),
        xn(max_tok * 2048),
        proj(max_tok * DeltaNetScratch::kProj),
        o(max_tok * 4096) {}
  GemmInput in;
  DeviceBuffer<float> xn, proj, o;
  float* qkv() { return proj.data(); }
  float* z(unsigned t) { return proj.data() + std::size_t{t} * DeltaNetScratch::kProj + 8192; }
};

// h += delta (if delta != nullptr), then y = DeltaNet(RMSNorm(h)) for one token. h, delta and y are
// device pointers to 2048 floats.
void deltanet_decode(const DeltaNetLayer& w, DeltaNetState& st, DeltaNetScratch& sc, float* h,
                     const float* delta, float* y, float eps, hipStream_t stream);

// The same for n consecutive tokens ([n][2048] arrays): projections as prefill GEMMs, the
// recurrence token by token in one launch. The chunked form (kernels/deltanet_chunk.hpp) matches
// this recurrence but is slower on gfx906, so prefill does not use it.
void deltanet_prefill(const DeltaNetLayer& w, DeltaNetState& st, DeltaNetPrefillScratch& sc,
                      float* h, const float* delta, float* y, unsigned n, float eps,
                      hipStream_t stream);

}  // namespace miso
