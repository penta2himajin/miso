#pragma once

// Decode GEMV y = W x for Q4_K weights in the r1 layout (ADR_001 D3 decode family, src/repack.hpp).
//
// One wavefront computes one row at a time. A "slot" is 16 bytes of a block's nibbles = 32 weights:
// 16 of sub-block 2g (low nibbles) and 16 of sub-block 2g+1 (high nibbles). A K=2048 row is 64
// slots, so each lane issues one 16-byte load per row. A lane's activations are loaded and
// converted once per wave and reused for every row the wave processes.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "wave.hpp"

namespace miso::kernels {

enum class ActFormat {
  Int8,  // per-32 absmax INT8 activations, v_dot4_i32_i8
  Fp16,  // FP16 activations, v_dot2_f32_f16
};

struct Q4kGemvParams {
  const std::uint8_t* w;  // n_rows rows of K/256 Q4_K r1 blocks; rows 16-byte aligned
  const float* x;         // K activations
  float* y;               // n_rows outputs
  unsigned n_rows;
};

namespace q4k_detail {

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
constexpr int kBlockBytes = 144;

__device__ inline float fp16_bits(unsigned h) {
  return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(h)));
}

struct Scales {
  float sc0, m0, sc1, m1;
};

// The r1 24-bit field of nibble group g starts at bit 24g of header words (y, z, w). Selecting
// between computed values, not between h.y / h.z / h.w, keeps the compiler from turning the choice
// into a dynamic offset into a stack copy of h (which cost LDS and scratch).
__device__ inline Scales r1_scales(const uint4& h, int g) {
  const unsigned v01 = __builtin_amdgcn_alignbit(h.z, h.y, g == 0 ? 0u : 24u);
  const unsigned v23 = g == 2 ? __builtin_amdgcn_alignbit(h.w, h.z, 16u) : (h.w >> 8);
  const unsigned v = g < 2 ? v01 : v23;
  return {static_cast<float>(v & 63u), static_cast<float>((v >> 6) & 63u),
          static_cast<float>((v >> 12) & 63u), static_cast<float>((v >> 18) & 63u)};
}

// Nibble pair (bits [s, s+4) and [s+16, s+20)) as FP16 subnormals (q 2^-24, q' 2^-24), exact.
// Unlike a biased (1024 + q) encoding, products stay proportional to q x, so the FP32 chain has
// no offset to cancel (that cancellation cost ~2e-3 relative error on real activations).
__device__ inline half2_t nibbles_sub(unsigned w, int s) {
  return __builtin_bit_cast(half2_t, (w >> s) & 0x000F000Fu);
}

// Activations of one slot, already converted.
template <ActFormat A>
struct SlotAct;

template <>
struct SlotAct<ActFormat::Int8> {
  unsigned q_lo[4], q_hi[4];  // 16 INT8 each, natural order
  float d_lo, d_hi, s_lo, s_hi;
};

template <>
struct SlotAct<ActFormat::Fp16> {
  half2_t a_lo[4], b_lo[4], a_hi[4], b_hi[4];  // (x[4c], x[4c+2]) and (x[4c+1], x[4c+3])
  float s_lo, s_hi;
};

__device__ inline void load16(const float* p, float* v) {
  for (int c = 0; c < 4; ++c) {
    const float4 f = reinterpret_cast<const float4*>(p)[c];
    v[4 * c] = f.x, v[4 * c + 1] = f.y, v[4 * c + 2] = f.z, v[4 * c + 3] = f.w;
  }
}

__device__ inline unsigned pack_int8(const float* v, float inv) {
  unsigned r = 0;
  for (int t = 0; t < 4; ++t) {
    const int q = static_cast<int>(__builtin_rintf(v[t] * inv));
    r |= (static_cast<unsigned>(q) & 0xFFu) << (8 * t);
  }
  return r;
}

__device__ inline half2_t to_half2(float a, float b) {
  return half2_t{static_cast<_Float16>(a), static_cast<_Float16>(b)};
}

template <ActFormat A>
__device__ inline SlotAct<A> load_slot(const float* x, int slot) {
  const int b = slot >> 3, i = slot & 7;
  const float* lo_p = x + 256 * b + 64 * (i >> 1) + 16 * (i & 1);
  float lo[16], hi[16];
  load16(lo_p, lo);
  load16(lo_p + 32, hi);
  // s_lo / s_hi feed the dmin * m term and must be sums of the *converted* activations: with exact
  // sums, the q term's rounding error scales with d * sc * q instead of |W| = |d sc q - dmin m|,
  // which broke the error bound on rows where the two terms cancel.
  SlotAct<A> s;
  s.s_lo = s.s_hi = 0;
  if constexpr (A == ActFormat::Int8) {
    // A sub-block's 32 activations are split between this lane and lane ^ 1.
    float m_lo = 0, m_hi = 0;
    for (int t = 0; t < 16; ++t)
      m_lo = fmaxf(m_lo, fabsf(lo[t])), m_hi = fmaxf(m_hi, fabsf(hi[t]));
    m_lo = fmaxf(m_lo, __shfl_xor(m_lo, 1));
    m_hi = fmaxf(m_hi, __shfl_xor(m_hi, 1));
    s.d_lo = m_lo / 127.0f, s.d_hi = m_hi / 127.0f;
    const float inv_lo = m_lo > 0 ? 127.0f / m_lo : 0.0f, inv_hi = m_hi > 0 ? 127.0f / m_hi : 0.0f;
    for (int c = 0; c < 4; ++c) {
      s.q_lo[c] = pack_int8(lo + 4 * c, inv_lo);
      s.q_hi[c] = pack_int8(hi + 4 * c, inv_hi);
    }
    int q_lo = 0, q_hi = 0;
    for (int c = 0; c < 4; ++c) {
      q_lo = __builtin_amdgcn_sdot4(static_cast<int>(s.q_lo[c]), 0x01010101, q_lo, false);
      q_hi = __builtin_amdgcn_sdot4(static_cast<int>(s.q_hi[c]), 0x01010101, q_hi, false);
    }
    s.s_lo = s.d_lo * static_cast<float>(q_lo), s.s_hi = s.d_hi * static_cast<float>(q_hi);
  } else {
    for (int c = 0; c < 4; ++c) {
      s.a_lo[c] = to_half2(lo[4 * c], lo[4 * c + 2]);
      s.b_lo[c] = to_half2(lo[4 * c + 1], lo[4 * c + 3]);
      s.a_hi[c] = to_half2(hi[4 * c], hi[4 * c + 2]);
      s.b_hi[c] = to_half2(hi[4 * c + 1], hi[4 * c + 3]);
      s.s_lo += static_cast<float>(s.a_lo[c].x) + static_cast<float>(s.a_lo[c].y) +
                static_cast<float>(s.b_lo[c].x) + static_cast<float>(s.b_lo[c].y);
      s.s_hi += static_cast<float>(s.a_hi[c].x) + static_cast<float>(s.a_hi[c].y) +
                static_cast<float>(s.b_hi[c].x) + static_cast<float>(s.b_hi[c].y);
    }
  }
  return s;
}

// Weights of one slot: the block header and the lane's 16 nibble bytes.
struct SlotW {
  uint4 h, q;
};

__device__ inline SlotW load_w(const std::uint8_t* row, int slot) {
  const std::uint8_t* blk = row + kBlockBytes * (slot >> 3);
  return {*reinterpret_cast<const uint4*>(blk),
          *reinterpret_cast<const uint4*>(blk + 16 + 16 * (slot & 7))};
}

// Partial dot product of one slot of one row.
template <ActFormat A>
__device__ inline float slot_dot(const SlotW& w, int slot, const SlotAct<A>& s) {
  const int i = slot & 7;
  const uint4& h = w.h;
  const unsigned qw[4] = {w.q.x, w.q.y, w.q.z, w.q.w};
  const float d = fp16_bits(h.x & 0xFFFFu), dmin = fp16_bits(h.x >> 16);
  const Scales sc = r1_scales(h, i >> 1);
  const float mins = dmin * (sc.m0 * s.s_lo + sc.m1 * s.s_hi);
  if constexpr (A == ActFormat::Int8) {
    int lo = 0, hi = 0;
    for (int c = 0; c < 4; ++c) {
      lo = __builtin_amdgcn_sdot4(static_cast<int>(qw[c] & 0x0F0F0F0Fu),
                                  static_cast<int>(s.q_lo[c]), lo, false);
      hi = __builtin_amdgcn_sdot4(static_cast<int>((qw[c] >> 4) & 0x0F0F0F0Fu),
                                  static_cast<int>(s.q_hi[c]), hi, false);
    }
    return d * (sc.sc0 * s.d_lo * static_cast<float>(lo) +
                sc.sc1 * s.d_hi * static_cast<float>(hi)) -
           mins;
  } else {
    float lo = 0, hi = 0;
    for (int c = 0; c < 4; ++c) {
      lo = __builtin_amdgcn_fdot2(nibbles_sub(qw[c], 0), s.a_lo[c], lo, false);
      lo = __builtin_amdgcn_fdot2(nibbles_sub(qw[c], 8), s.b_lo[c], lo, false);
      hi = __builtin_amdgcn_fdot2(nibbles_sub(qw[c], 4), s.a_hi[c], hi, false);
      hi = __builtin_amdgcn_fdot2(nibbles_sub(qw[c], 12), s.b_hi[c], hi, false);
    }
    return d * 0x1p24f * (sc.sc0 * lo + sc.sc1 * hi) - mins;
  }
}

}  // namespace q4k_detail

// Rows [first, first + count). Each wavefront takes kRows consecutive rows per step, issuing all of
// their loads before computing, and strides over the grid's wavefronts.
template <ActFormat A, unsigned K, int kBlock, int kRows = 1>
__device__ void q4k_gemv_op(const Q4kGemvParams& p, unsigned first, unsigned count) {
  using namespace q4k_detail;
  static_assert(K % (32 * kWave) == 0, "K must be a multiple of 2048");
  static_assert(kBlock % kWave == 0);
  constexpr int kIters = K / (32 * kWave);
  constexpr unsigned kRowBytes = K / 256 * kBlockBytes;
  const int lane = threadIdx.x % kWave;
  const unsigned wave = blockIdx.x * (kBlock / kWave) + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * (kBlock / kWave);

  SlotAct<A> act[kIters];
  for (int it = 0; it < kIters; ++it)
    act[it] = load_slot<A>(p.x, lane + kWave * it);

  const unsigned end = first + count;
  for (unsigned r0 = first + wave * kRows; r0 < end; r0 += n_waves * kRows) {
    // Fully unrolled with constant indices so w stays in VGPRs (a data-dependent exit here made the
    // compiler move the array to LDS / scratch).
    SlotW w[kRows][kIters];
#pragma unroll
    for (int rr = 0; rr < kRows; ++rr) {
      // Rows past the end reload the last valid row; their results are not stored.
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
        acc += slot_dot<A>(w[rr][it], lane + kWave * it, act[it]);
      acc = wave_sum(acc);
      if (lane == kWave - 1 && r0 + rr < end)
        p.y[r0 + rr] = acc;
    }
  }
}

template <ActFormat A, unsigned K, int kBlock, int kRows = 1>
__global__ void __launch_bounds__(kBlock) q4k_gemv_kernel(Q4kGemvParams p) {
  q4k_gemv_op<A, K, kBlock, kRows>(p, 0, p.n_rows);
}

}  // namespace miso::kernels
