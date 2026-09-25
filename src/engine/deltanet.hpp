#pragma once

// Gated DeltaNet token mixer of one layer, decode path (one token per call).

#include <hip/hip_runtime.h>

#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

namespace miso {

struct DeltaNetLayer {
  QMatrix qkv, z, a, b, out;  // attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out
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

struct DeltaNetScratch {
  DeviceBuffer<float> xn{2048}, qkv{8192}, z{4096}, a{32}, b{32}, o{4096};
};

// y = DeltaNet(RMSNorm(h)) for one token; h and y are device pointers to 2048 floats.
void deltanet_decode(const DeltaNetLayer& w, DeltaNetState& st, DeltaNetScratch& sc, const float* h,
                     float* y, float eps, hipStream_t stream);

}  // namespace miso
