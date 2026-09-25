#pragma once

// Decode GEMV y = W x for Q4_K weights in GGUF layout (ADR_001 D3, decode family).
//
// One wavefront computes one row at a time. A "slot" is 16 bytes of a block's nibbles = 32 weights:
// 16 of sub-block 2g (low nibbles) and 16 of sub-block 2g+1 (high nibbles). A K=2048 row is 64
// slots, so each lane issues one 16-byte load per row. A lane's activations are loaded and
// converted once per wave and reused for every row the wave processes.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace miso::kernels {

enum class ActFormat {
  Int8,  // per-32 absmax INT8 activations, v_dot4_i32_i8
  Fp16,  // FP16 activations, v_dot2_f32_f16
};

struct Q4kGemvParams {
  const std::uint8_t* w;  // n_rows rows of K/256 Q4_K blocks; rows 16-byte aligned
  const float* x;         // K activations
  float* y;               // n_rows outputs
  unsigned n_rows;
};

namespace q4k_detail {

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
constexpr int kWave = 64;
constexpr int kBlockBytes = 144;

__device__ inline float fp16_bits(unsigned h) {
  return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(h)));
}

// Byte k (0..11) of the packed scales, which occupy header words y, z, w.
__device__ inline unsigned scale_byte(const uint4& h, int k) {
  const unsigned w = k < 4 ? h.y : (k < 8 ? h.z : h.w);
  return (w >> (8 * (k & 3))) & 0xFFu;
}

// Same unpacking as the CPU reference (src/dequant.cpp).
__device__ inline void scale_min(const uint4& h, int j, float& sc, float& m) {
  if (j < 4) {
    sc = static_cast<float>(scale_byte(h, j) & 63u);
    m = static_cast<float>(scale_byte(h, j + 4) & 63u);
  } else {
    sc = static_cast<float>((scale_byte(h, j + 4) & 15u) | ((scale_byte(h, j - 4) >> 2) & 0x30u));
    m = static_cast<float>((scale_byte(h, j + 4) >> 4) | ((scale_byte(h, j) >> 2) & 0x30u));
  }
}

// Nibble pair (bits [s, s+4) and [s+16, s+20)) as an exact FP16 pair: 0x6400|q is 1024+q.
__device__ inline half2_t nibbles_to_half2(unsigned w, int s) {
  const unsigned bits = ((w >> s) & 0x000F000Fu) | 0x64006400u;
  return __builtin_bit_cast(half2_t, bits) - half2_t{1024, 1024};
}

__device__ inline float wave_sum(float v) {
  for (int off = kWave / 2; off > 0; off >>= 1)
    v += __shfl_xor(v, off);
  return v;
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

template <ActFormat A>
__device__ inline SlotAct<A> load_slot(const float* x, int slot) {
  const int b = slot >> 3, i = slot & 7;
  const float* lo_p = x + 256 * b + 64 * (i >> 1) + 16 * (i & 1);
  float lo[16], hi[16];
  load16(lo_p, lo);
  load16(lo_p + 32, hi);
  SlotAct<A> s;
  s.s_lo = s.s_hi = 0;
  for (int t = 0; t < 16; ++t)
    s.s_lo += lo[t], s.s_hi += hi[t];
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
  } else {
    for (int c = 0; c < 4; ++c) {
      s.a_lo[c] = half2_t{static_cast<_Float16>(lo[4 * c]), static_cast<_Float16>(lo[4 * c + 2])};
      s.b_lo[c] =
          half2_t{static_cast<_Float16>(lo[4 * c + 1]), static_cast<_Float16>(lo[4 * c + 3])};
      s.a_hi[c] = half2_t{static_cast<_Float16>(hi[4 * c]), static_cast<_Float16>(hi[4 * c + 2])};
      s.b_hi[c] =
          half2_t{static_cast<_Float16>(hi[4 * c + 1]), static_cast<_Float16>(hi[4 * c + 3])};
    }
  }
  return s;
}

// Partial dot product of one slot of one row.
template <ActFormat A>
__device__ inline float slot_dot(const std::uint8_t* row, int slot, const SlotAct<A>& s) {
  const int b = slot >> 3, i = slot & 7;
  const std::uint8_t* blk = row + kBlockBytes * b;
  const uint4 h = *reinterpret_cast<const uint4*>(blk);
  const uint4 q = *reinterpret_cast<const uint4*>(blk + 16 + 16 * i);
  const unsigned qw[4] = {q.x, q.y, q.z, q.w};
  const float d = fp16_bits(h.x & 0xFFFFu), dmin = fp16_bits(h.x >> 16);
  float sc0, m0, sc1, m1;
  scale_min(h, 2 * (i >> 1), sc0, m0);
  scale_min(h, 2 * (i >> 1) + 1, sc1, m1);
  const float mins = dmin * (m0 * s.s_lo + m1 * s.s_hi);
  if constexpr (A == ActFormat::Int8) {
    int lo = 0, hi = 0;
    for (int c = 0; c < 4; ++c) {
      lo = __builtin_amdgcn_sdot4(static_cast<int>(qw[c] & 0x0F0F0F0Fu),
                                  static_cast<int>(s.q_lo[c]), lo, false);
      hi = __builtin_amdgcn_sdot4(static_cast<int>((qw[c] >> 4) & 0x0F0F0F0Fu),
                                  static_cast<int>(s.q_hi[c]), hi, false);
    }
    return d * (sc0 * s.d_lo * static_cast<float>(lo) + sc1 * s.d_hi * static_cast<float>(hi)) -
           mins;
  } else {
    float lo = 0, hi = 0;
    for (int c = 0; c < 4; ++c) {
      lo = __builtin_amdgcn_fdot2(nibbles_to_half2(qw[c], 0), s.a_lo[c], lo, false);
      lo = __builtin_amdgcn_fdot2(nibbles_to_half2(qw[c], 8), s.b_lo[c], lo, false);
      hi = __builtin_amdgcn_fdot2(nibbles_to_half2(qw[c], 4), s.a_hi[c], hi, false);
      hi = __builtin_amdgcn_fdot2(nibbles_to_half2(qw[c], 12), s.b_hi[c], hi, false);
    }
    return d * (sc0 * lo + sc1 * hi) - mins;
  }
}

}  // namespace q4k_detail

// Rows [first, first + count), one row per wavefront at a time (grid-stride over waves).
template <ActFormat A, unsigned K, int kBlock>
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

  for (unsigned r = first + wave; r < first + count; r += n_waves) {
    const std::uint8_t* row = p.w + std::size_t{r} * kRowBytes;
    float acc = 0;
    for (int it = 0; it < kIters; ++it)
      acc += slot_dot<A>(row, lane + kWave * it, act[it]);
    acc = wave_sum(acc);
    if (lane == 0)
      p.y[r] = acc;
  }
}

template <ActFormat A, unsigned K, int kBlock>
__global__ void __launch_bounds__(kBlock) q4k_gemv_kernel(Q4kGemvParams p) {
  q4k_gemv_op<A, K, kBlock>(p, 0, p.n_rows);
}

}  // namespace miso::kernels
