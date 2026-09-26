#pragma once

// Frozen FP32-LDS reference from main d3dd1a6, for exact staging comparisons.
// Test-only: preserve its activation conversion, expert reduction and arithmetic order.
#include "moe_decode.hpp"

namespace miso::kernels {

template <QType T, int kBlock>
__device__ void fp32_down_reference_op(const MoeDownParams& p, unsigned first, unsigned count) {
  using D = moe_detail::Down<T>;
  constexpr int kSteps = (moe::kSlots + 3) / 4;
  __shared__ float a[moe::kSlots][moe::kFf];
  __shared__ int sid[moe::kTopK];
  __shared__ float coef[moe::kSlots];
  for (unsigned i = threadIdx.x; i < moe::kSlots * moe::kFf; i += kBlock)
    a[i / moe::kFf][i % moe::kFf] = p.h[i];
  if (threadIdx.x < moe::kTopK)
    sid[threadIdx.x] = p.ids[threadIdx.x], coef[threadIdx.x] = p.weights[threadIdx.x];
  if (threadIdx.x == 0)
    coef[moe::kTopK] = 1.0f / (1.0f + expf(-p.logits[moe::kExperts]));
  __syncthreads();

  const int lane = threadIdx.x % kWave, grp = lane / 16, l16 = lane % 16;
  const unsigned wave = blockIdx.x * (kBlock / kWave) + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * (kBlock / kWave);
  typename D::Act act[kSteps];
#pragma unroll
  for (int it = 0; it < kSteps; ++it) {
    const int s = min(4 * it + grp, moe::kSlots - 1);
    act[it] = D::act(a[s], l16);
  }
  for (unsigned r = first + wave; r < first + count; r += n_waves) {
    float v = 0;
#pragma unroll
    for (int it = 0; it < kSteps; ++it) {
      const int s = 4 * it + grp, sc = min(s, moe::kSlots - 1);
      const std::uint8_t* base =
          sc < moe::kTopK ? p.down_exps + (std::size_t(sid[sc]) * moe::kHidden + r) * D::kRowBytes
                          : p.down_sh + std::size_t{r} * D::kRowBytes;
      const float part = moe_detail::row16_sum(D::dot(base, l16, act[it]));
      v += s < moe::kSlots ? coef[sc] * part : 0.0f;
    }
    v += dpp<0x142, 0xA>(v);  // rows 1, 3 += rows 0, 2
    v += dpp<0x143, 0xC>(v);  // rows 2, 3 += row 1: lane 63 holds the total
    if (lane == kWave - 1)
      p.out[r] = v;
  }
}

template <QType T, int kBlock>
__global__ void __launch_bounds__(kBlock) fp32_down_reference_kernel(MoeDownParams p) {
  fp32_down_reference_op<T, kBlock>(p, 0, moe::kHidden);
}

}  // namespace miso::kernels
