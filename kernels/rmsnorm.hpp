#pragma once

// Decode residual add + RMSNorm: residual += delta; out = residual * rsqrt(mean(residual^2) + eps)
// * weight. `weight` is the GGUF tensor, which already stores 1 + w
// (docs/research/ornith-q4km-gguf.md).

#include <hip/hip_runtime.h>

#include "wave.hpp"

namespace miso::kernels {

// Decode add+RMSNorm workgroup size. 512 beat 256 end-to-end on MI50 (run18: median 141.5 vs
// 136.3 tok/s); 128 regressed. kN=2048 requires a multiple of kWave that divides 2048.
inline constexpr int kRmsNormBlock = 512;

struct AddRmsNormParams {
  float* residual;      // [tokens][kN], updated in place
  const float* delta;   // [tokens][kN], or nullptr for a plain RMSNorm
  const float* weight;  // [kN]
  float* out;           // [tokens][kN]
  float eps;
};

// Tokens [first, first + count); one workgroup per token (grid-stride).
template <unsigned kN, int kBlock>
__device__ void add_rmsnorm_op(const AddRmsNormParams& p, unsigned first, unsigned count) {
  static_assert(kN % kBlock == 0);
  constexpr int kPer = kN / kBlock;
  __shared__ float scratch[kBlock / kWave];
  for (unsigned t = first + blockIdx.x; t < first + count; t += gridDim.x) {
    float* res = p.residual + std::size_t{t} * kN;
    float h[kPer];
    float ss = 0;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const unsigned i = threadIdx.x + k * kBlock;
      h[k] = p.delta != nullptr ? res[i] + p.delta[std::size_t{t} * kN + i] : res[i];
      ss += h[k] * h[k];
    }
    const float r = rsqrtf(block_sum<kBlock>(ss, scratch) / kN + p.eps);
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const unsigned i = threadIdx.x + k * kBlock;
      if (p.delta != nullptr)
        res[i] = h[k];
      p.out[std::size_t{t} * kN + i] = h[k] * r * p.weight[i];
    }
  }
}

template <unsigned kN, int kBlock>
__global__ void __launch_bounds__(kBlock) add_rmsnorm_kernel(AddRmsNormParams p, unsigned tokens) {
  add_rmsnorm_op<kN, kBlock>(p, 0, tokens);
}

}  // namespace miso::kernels
