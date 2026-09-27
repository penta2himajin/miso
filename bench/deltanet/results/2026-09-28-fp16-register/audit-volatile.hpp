#pragma once

// Benchmark-local activation preparation; production kernels and dispatch stay unchanged.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "deltanet_step.hpp"
#include "gemv_common.hpp"
#include "q4k_gemv.hpp"

namespace miso::bench::audit_volatile {

enum class Output { Fp32, Dual, Fp16 };

struct StepParams {
  kernels::DeltaNetStepParams base;
  _Float16* half_out;
};

namespace frozen_detail {

// Frozen af54ec7 scalar helpers: producer changes only its final output stores.
__device__ inline float silu(float x) {
  return x / (1.0f + __expf(-x));
}
__device__ inline float softplus(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ inline float conv_step(const kernels::DeltaNetStepParams& p, unsigned c, bool own) {
  const float* s = p.conv_in + 3 * c;
  const float* w = p.conv_w + 4 * c;
  const float x = p.qkv[c];
  const float y = w[0] * s[0] + w[1] * s[1] + w[2] * s[2] + w[3] * x;
  if (own) {
    p.conv_out[3 * c] = s[1];
    p.conv_out[3 * c + 1] = s[2];
    p.conv_out[3 * c + 2] = x;
  }
  return silu(y);
}

}  // namespace frozen_detail

template <Output Mode>
__device__ void step_op(const StepParams& params, unsigned first, unsigned count) {
  using namespace kernels;
  using namespace frozen_detail;
  constexpr int kDim = 128, kKHeads = 16, kBlock = 256;
  const auto& p = params.base;
  __shared__ float q[kDim], k[kDim], v[kDim], red[kBlock / kWave], part[2][kDim];
  const unsigned t = threadIdx.x, vi = t % kDim, half = t / kDim, wave = t / kWave;
  for (unsigned h = first + blockIdx.x; h < first + count; h += gridDim.x) {
    const unsigned kh = h % kKHeads;
    const bool own_qk = h < kKHeads;
    if (t < kDim) {
      q[t] = conv_step(p, kh * kDim + t, own_qk);
      v[t] = conv_step(p, 2 * kKHeads * kDim + h * kDim + t, true);
    } else {
      k[vi] = conv_step(p, kKHeads * kDim + kh * kDim + vi, own_qk);
    }
    __syncthreads();
    const float x = t < kDim ? q[t] : k[vi];
    const float ss = wave_sum(x * x);
    if (t % kWave == kWave - 1)
      red[wave] = ss;
    __syncthreads();
    if (t < kDim) {
      q[t] = x * rsqrtf(red[0] + red[1] + 1e-6f) * rsqrtf(static_cast<float>(kDim));
    } else {
      k[vi] = x * rsqrtf(red[2] + red[3] + 1e-6f);
    }
    __syncthreads();
    const float beta = 1.0f / (1.0f + __expf(-p.b[h]));
    const float decay = __expf(p.ssm_a[h] * softplus(p.a[h] + p.dt_bias[h]));
    float* S = p.state + std::size_t{h} * kDim * kDim;
    float s[kDim / 2];
    float kv = 0;
#pragma unroll
    for (int j = 0; j < kDim / 2; ++j) {
      const unsigned kk = half * (kDim / 2) + j;
      s[j] = S[kk * kDim + vi] * decay;
      kv += s[j] * k[kk];
    }
    part[half][vi] = kv;
    __syncthreads();
    const float delta = (v[vi] - (part[0][vi] + part[1][vi])) * beta;
    __syncthreads();
    float o = 0;
#pragma unroll
    for (int j = 0; j < kDim / 2; ++j) {
      const unsigned kk = half * (kDim / 2) + j;
      s[j] += k[kk] * delta;
      S[kk * kDim + vi] = s[j];
      o += s[j] * q[kk];
    }
    part[half][vi] = o;
    __syncthreads();
    const float oh = t < kDim ? part[0][t] + part[1][t] : 0.0f;
    const float os = wave_sum(oh * oh);
    if (t % kWave == kWave - 1)
      red[wave] = os;
    __syncthreads();
    if (t < kDim) {
      const float r = rsqrtf((red[0] + red[1]) / kDim + p.eps);
      const float value = oh * r * p.norm_w[t] * silu(p.z[h * kDim + t]);
      // Finalize FP32 bits before rounding. A live-register cast can differ by 1 ULP from
      // casting the stored float on gfx906 (token 7, index 2647).
      if constexpr (Mode == Output::Fp32) {
        p.out[h * kDim + t] = value;
      } else if constexpr (Mode == Output::Dual) {
        volatile float* out = &p.out[h * kDim + t];
        *out = value;
        params.half_out[h * kDim + t] = static_cast<_Float16>(*out);
      } else {
        volatile float tmp = value;
        params.half_out[h * kDim + t] = static_cast<_Float16>(tmp);
      }
    }
    __syncthreads();
  }
}

template <Output Mode>
__global__ void __launch_bounds__(256) step_kernel(StepParams p) {
  step_op<Mode>(p, 0, 32);
}

struct HalfGemvParams {
  const std::uint8_t* w;
  const _Float16* x;
  float* y;
  unsigned n_rows;
};

__device__ inline kernels::q4k_detail::SlotAct<kernels::ActFormat::Fp16> load_half_slot(
    const _Float16* x, int slot) {
  const int b = slot >> 3, i = slot & 7;
  const _Float16* lo_p = x + 256 * b + 64 * (i >> 1) + 16 * (i & 1);
  const auto lo = kernels::gemv::fp16x16(lo_p), hi = kernels::gemv::fp16x16(lo_p + 32);
  kernels::q4k_detail::SlotAct<kernels::ActFormat::Fp16> s;
  s.s_lo = lo.s16;
  s.s_hi = hi.s16;
  for (int c = 0; c < 4; ++c) {
    s.a_lo[c] = lo.a[c];
    s.b_lo[c] = lo.b[c];
    s.a_hi[c] = hi.a[c];
    s.b_hi[c] = hi.b[c];
  }
  return s;
}

// Frozen af54ec7 R4 row/weight pipeline and per-slot dot/sum order; only the activation reader
// changes. Slot sums use already-rounded halves in the original nibble-pair order.
template <unsigned K, int kBlock, int kRows>
__device__ void q4_half_gemv_op(const HalfGemvParams& p, unsigned first, unsigned count) {
  using namespace kernels;
  using namespace q4k_detail;
  static_assert(K % (32 * kWave) == 0);
  static_assert(kBlock % kWave == 0);
  constexpr int kIters = K / (32 * kWave);
  constexpr unsigned kRowBytes = K / 256 * kBlockBytes;
  const int lane = threadIdx.x % kWave;
  const unsigned wave = blockIdx.x * (kBlock / kWave) + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * (kBlock / kWave);
  SlotAct<ActFormat::Fp16> act[kIters];
  for (int it = 0; it < kIters; ++it)
    act[it] = load_half_slot(p.x, lane + kWave * it);
  const unsigned end = first + count;
  for (unsigned r0 = first + wave * kRows; r0 < end; r0 += n_waves * kRows) {
    SlotW w[kRows][kIters];
#pragma unroll
    for (int rr = 0; rr < kRows; ++rr) {
      const unsigned r = r0 + rr < end ? r0 + rr : end - 1;
      const std::uint8_t* row = p.w + std::size_t{r} * kRowBytes;
#pragma unroll
      for (int it = 0; it < kIters; ++it)
        w[rr][it] = load_w(row, lane + kWave * it);
    }
#pragma unroll
    for (int rr = 0; rr < kRows; ++rr) {
      float acc = 0;
#pragma unroll
      for (int it = 0; it < kIters; ++it)
        acc += slot_dot<ActFormat::Fp16>(w[rr][it], lane + kWave * it, act[it]);
      acc = wave_sum(acc);
      if (lane == kWave - 1 && r0 + rr < end)
        p.y[r0 + rr] = acc;
    }
  }
}

template <unsigned K, int kBlock, int kRows = 4>
__global__ void __launch_bounds__(kBlock) q4_half_gemv_kernel(HalfGemvParams p) {
  q4_half_gemv_op<K, kBlock, kRows>(p, 0, p.n_rows);
}

}  // namespace miso::bench::audit_volatile
