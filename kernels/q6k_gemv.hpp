#pragma once

// Decode GEMV y = W x for Q6_K weights in the r1 row layout (src/repack.hpp), FP16 activations
// (ADR_004).
//
// Eight lanes cover one 256-weight block. Lane i takes half h = i / 4 and column c = 16 (i % 4) of
// that half. It reads 16 ql bytes (low nibbles -> elements 128h + c + t, high nibbles ->
// 128h + 64 + c + t, t < 16) and the 16 qh bytes carrying their top two bits. Each group of 16 has
// one int8 scale. A K=2048 row is 64 lanes x 32 weights.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "gemv_common.hpp"
#include "wave.hpp"

namespace miso::kernels {

struct Q6kGemvParams {
  const std::uint8_t* w;  // n_rows Q6_K r1 rows of K / 256 blocks
  const float* x;         // K activations
  float* y;               // n_rows outputs
  unsigned n_rows;
};

namespace q6k_detail {

using gemv::half2_t;

// (q6 2^-24, q6' 2^-24) as FP16 subnormals for the two elements at bits [s, s+4) / [s+16, s+20) of
// `ql` and [t, t+2) / [t+16, t+18) of `qh`; q6 = q + 32 is the unsigned 6-bit code. Exact.
__device__ inline half2_t q6_sub(unsigned ql, int s, unsigned qh, int t) {
  return __builtin_bit_cast(half2_t, ((ql >> s) & 0x000F000Fu) | (((qh >> t) & 0x00030003u) << 4));
}

struct SlotAct {
  gemv::Fp16x16 lo, hi;
};

// Activations of one slot: elements 128h + c + t (lo) and 128h + 64 + c + t (hi) of its block.
__device__ inline SlotAct load_slot(const float* x, int slot) {
  const int b = slot >> 3, i = slot & 7;
  const float* lo = x + 256 * b + 128 * (i >> 2) + 16 * (i & 3);
  return {gemv::fp16x16(lo), gemv::fp16x16(lo + 64)};
}

struct SlotW {
  uint4 ql, qh;
  uint2 sc;
  unsigned short d;
};

template <unsigned K>
__device__ inline SlotW load_w(const std::uint8_t* row, int slot) {
  constexpr unsigned nb = K / 256;
  const int b = slot >> 3, i = slot & 7, h = i >> 2, c = 16 * (i & 3);
  return {*reinterpret_cast<const uint4*>(row + 128 * b + 64 * h + c),
          *reinterpret_cast<const uint4*>(row + nb * 128 + 64 * b + 32 * h + (c & 31)),
          *reinterpret_cast<const uint2*>(row + nb * 192 + 16 * b + 8 * h),
          *reinterpret_cast<const unsigned short*>(row + nb * 208 + 2 * b)};
}

__device__ inline float slot_dot(const SlotW& w, int slot, const SlotAct& a) {
  const int i = slot & 7, c = 16 * (i & 3);
  const int t_lo = 2 * (c >> 5), t_hi = t_lo + 4;
  const unsigned ql[4] = {w.ql.x, w.ql.y, w.ql.z, w.ql.w};
  const unsigned qh[4] = {w.qh.x, w.qh.y, w.qh.z, w.qh.w};
  const float sc_lo = static_cast<float>(static_cast<std::int8_t>(w.sc.x >> (8 * (i & 3))));
  const float sc_hi = static_cast<float>(static_cast<std::int8_t>(w.sc.y >> (8 * (i & 3))));
  float lo = -32.0f * 0x1p-24f * a.lo.s16, hi = -32.0f * 0x1p-24f * a.hi.s16;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    lo = __builtin_amdgcn_fdot2(q6_sub(ql[k], 0, qh[k], t_lo), a.lo.a[k], lo, false);
    lo = __builtin_amdgcn_fdot2(q6_sub(ql[k], 8, qh[k], t_lo + 8), a.lo.b[k], lo, false);
    hi = __builtin_amdgcn_fdot2(q6_sub(ql[k], 4, qh[k], t_hi), a.hi.a[k], hi, false);
    hi = __builtin_amdgcn_fdot2(q6_sub(ql[k], 12, qh[k], t_hi + 8), a.hi.b[k], hi, false);
  }
  return gemv::fp16_bits(w.d) * 0x1p24f * (sc_lo * lo + sc_hi * hi);
}

}  // namespace q6k_detail

// Rows [first, first + count); kRows consecutive rows per wavefront step, loads issued first.
template <unsigned K, int kBlock, int kRows = 1>
__device__ void q6k_gemv_op(const Q6kGemvParams& p, unsigned first, unsigned count) {
  using namespace q6k_detail;
  static_assert(K % (32 * kWave) == 0, "K must be a multiple of 2048");
  static_assert(kBlock % kWave == 0);
  constexpr int kIters = K / (32 * kWave);
  const std::size_t row_bytes = (K / 256 * 210 + 15) / 16 * 16;
  const int lane = threadIdx.x % kWave;
  const unsigned wave = blockIdx.x * (kBlock / kWave) + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * (kBlock / kWave);

  SlotAct act[kIters];
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    act[it] = load_slot(p.x, lane + kWave * it);
  }

  const unsigned end = first + count;
  for (unsigned r0 = first + wave * kRows; r0 < end; r0 += n_waves * kRows) {
    SlotW w[kRows][kIters];
#pragma unroll
    for (int rr = 0; rr < kRows; ++rr) {
      const unsigned r = r0 + rr < end ? r0 + rr : end - 1;
#pragma unroll
      for (int it = 0; it < kIters; ++it)
        w[rr][it] = load_w<K>(p.w + r * row_bytes, lane + kWave * it);
    }
#pragma unroll
    for (int rr = 0; rr < kRows; ++rr) {
      float acc = 0;
#pragma unroll
      for (int it = 0; it < kIters; ++it)
        acc += slot_dot(w[rr][it], lane + kWave * it, act[it]);
      acc = wave_sum(acc);
      if (lane == kWave - 1 && r0 + rr < end)
        p.y[r0 + rr] = acc;
    }
  }
}

template <unsigned K, int kBlock, int kRows = 1>
__global__ void __launch_bounds__(kBlock) q6k_gemv_kernel(Q6kGemvParams p) {
  q6k_gemv_op<K, kBlock, kRows>(p, 0, p.n_rows);
}

}  // namespace miso::kernels
