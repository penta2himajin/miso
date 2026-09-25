#pragma once

// Prefill GEMM Y[t][n] = sum_k W[n][k] x[t][k] for Q4_K r1 / Q6_K r1 weights (ADR_001 D3 prefill
// family) with FP16 activations and FP32 accumulation (v_dot2_f32_f16).
//
// Weights are expanded once per workgroup into LDS as exact FP16 integers and reused by all BM
// tokens of the tile:
//   Q4_K: q * sc (<= 15 * 63, exact); per 256-block the partial is scaled by d, and the min term
//         dmin * m * sum(x) uses per-32 sums of the rounded activations (as in decode, ADR_004).
//   Q6_K: q - 32 (exact); per 16-group the partial is scaled by d * sc. (q - 32) * sc is not
//         always representable in FP16 (|sc| reaches 128), so the scale stays out of the weight.
// Activations are rounded to FP16 by qgemm_act_op, exactly like the decode GEMV does.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "q4k_gemv.hpp"

namespace miso::kernels {

enum class QFormat { Q4K, Q6K };

struct QGemmActParams {
  const float* x;  // n_tok x k
  _Float16* xh;    // n_tok x k, x rounded to FP16
  float* xs;       // n_tok x k / 32, sums of the rounded values of each 32-group
  unsigned k, n_tok;
};

struct QGemmParams {
  const std::uint8_t* w;  // n_rows rows in the format's r1 layout
  const _Float16* xh;     // n_tok x k
  const float* xs;        // n_tok x k / 32 (Q4_K only)
  float* y;               // n_tok rows of stride ldy
  unsigned k, n_rows, n_tok, ldy;
};

// One thread per 32-group: [first, first + count) over n_tok * k / 32 groups.
__device__ inline void qgemm_act_op(const QGemmActParams& p, unsigned first, unsigned count) {
  for (unsigned g = first + blockIdx.x * blockDim.x + threadIdx.x; g < first + count;
       g += gridDim.x * blockDim.x) {
    const float* x = p.x + std::size_t{g} * 32;
    _Float16* xh = p.xh + std::size_t{g} * 32;
    float s = 0;
    for (int c = 0; c < 8; ++c) {
      const float4 f = reinterpret_cast<const float4*>(x)[c];
      const _Float16 h[4] = {static_cast<_Float16>(f.x), static_cast<_Float16>(f.y),
                             static_cast<_Float16>(f.z), static_cast<_Float16>(f.w)};
      s += static_cast<float>(h[0]) + static_cast<float>(h[1]) + static_cast<float>(h[2]) +
           static_cast<float>(h[3]);
      *reinterpret_cast<uint2*>(xh + 4 * c) = __builtin_bit_cast(uint2, h);
    }
    p.xs[g] = s;
  }
}

__global__ void __launch_bounds__(256) qgemm_act_kernel(QGemmActParams p) {
  qgemm_act_op(p, 0, p.n_tok * (p.k / 32));
}

namespace qgemm_detail {

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

constexpr int kThreads = 256;
constexpr int kBM = 64, kBN = 64, kBK = 64;  // tokens, rows, K per stage
constexpr int kLd = kBK + 8;                 // LDS row stride in halves
constexpr int kTM = kBM / 16, kTN = kBN / 16;

struct alignas(16) Smem {
  _Float16 x[kBM][kLd];
  _Float16 w[kBN][kLd];
  float sc[kBN][4];  // Q4_K: d, dmin m0, dmin m1; Q6_K: d sc of the stage's four 16-groups
  float xs[kBM][2];  // Q4_K: sums of the stage's two 32-groups
};

__device__ inline half2_t h2(unsigned v) {
  return __builtin_bit_cast(half2_t, v);
}

__device__ inline unsigned u32(half2_t v) {
  return __builtin_bit_cast(unsigned, v);
}

// A pair of small integers v at bits [0, 10) and [16, 26) -> the FP16 pair (v - bias) * s. Both
// steps are exact while |(v - bias) * s| <= 2048.
__device__ inline unsigned int_pair(unsigned v, _Float16 bias, half2_t s) {
  return u32((h2(v | 0x64006400u) - half2_t{1024 + bias, 1024 + bias}) * s);
}

template <QFormat F>
constexpr unsigned row_bytes(unsigned k) {
  return F == QFormat::Q4K ? k / 256 * 144 : (k / 256 * 210 + 15) / 16 * 16;
}

// Global data one thread contributes to a stage (K offset 64 s): 16 weights of row tid / 4 and two
// 8-half activation chunks. Fetched a stage ahead so the loads overlap the previous stage's math.
template <QFormat F>
struct Fetch;

template <>
struct Fetch<QFormat::Q4K> {
  uint4 h;   // block header (d, dmin, r1 scales)
  uint2 qs;  // 8 qs bytes
  uint4 x[2];
  float2 xs;  // sums of the stage's two 32-groups (threads < kBM)
};

template <>
struct Fetch<QFormat::Q6K> {
  uint4 ql, qh;
  unsigned short d;
  std::int8_t sc;
  uint4 x[2];
};

template <QFormat F>
__device__ inline Fetch<F> fetch(const QGemmParams& p, unsigned n0, unsigned m0, unsigned s,
                                 int tid) {
  Fetch<F> f;
  const int r = tid >> 2, q = tid & 3;
  const unsigned gr = n0 + r < p.n_rows ? n0 + r : p.n_rows - 1;
  const std::uint8_t* row = p.w + std::size_t{gr} * row_bytes<F>(p.k);
  const unsigned b = s >> 2, ss = s & 3;
  if constexpr (F == QFormat::Q4K) {
    // Nibble group ss: qs bytes 32 ss + 8 q + i hold local k = 8 q + i (low) and 32 + 8 q + i.
    const std::uint8_t* blk = row + 144 * b;
    f.h = *reinterpret_cast<const uint4*>(blk);
    f.qs = *reinterpret_cast<const uint2*>(blk + 16 + 32 * ss + 8 * q);
    if (tid < kBM) {
      const unsigned t = m0 + tid;
      f.xs = t < p.n_tok
                 ? *reinterpret_cast<const float2*>(p.xs + std::size_t{t} * (p.k / 32) + 2 * s)
                 : float2{0, 0};
    }
  } else {
    // Local k = 16 q + i is element 128 hh + 64 u + 16 q + i of block b: ql byte l0 + i of quarter
    // `quad`'s 32-byte run (nibble u), qh byte l0 + i (bits 2 quad).
    const unsigned nb = p.k / 256, hh = ss >> 1, u = ss & 1, quad = 2 * u + (q >> 1);
    const unsigned l0 = 16 * (q & 1);
    f.ql = *reinterpret_cast<const uint4*>(row + 128 * b + 64 * hh + l0 + 32 * (quad & 1));
    f.qh = *reinterpret_cast<const uint4*>(row + nb * 128 + 64 * b + 32 * hh + l0);
    f.sc = static_cast<std::int8_t>(row[nb * 192 + 16 * b + 8 * hh + 4 * u + q]);
    f.d = *reinterpret_cast<const unsigned short*>(row + nb * 208 + 2 * b);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tid + kThreads * i, tok = idx >> 3, c = idx & 7;
    const unsigned t = m0 + tok;
    f.x[i] = t < p.n_tok
                 ? *reinterpret_cast<const uint4*>(p.xh + std::size_t{t} * p.k + kBK * s + 8 * c)
                 : uint4{0, 0, 0, 0};
  }
  return f;
}

// Expands a fetched stage into LDS.
template <QFormat F>
__device__ inline void store(const Fetch<F>& f, Smem& sm, unsigned s, int tid) {
  const int r = tid >> 2, q = tid & 3;
  const unsigned ss = s & 3;
  if constexpr (F == QFormat::Q4K) {
    const q4k_detail::Scales sc = q4k_detail::r1_scales(f.h, static_cast<int>(ss));
    const half2_t s0{static_cast<_Float16>(sc.sc0), static_cast<_Float16>(sc.sc0)};
    const half2_t s1{static_cast<_Float16>(sc.sc1), static_cast<_Float16>(sc.sc1)};
    unsigned lo[4], hi[4];
    const unsigned wq[2] = {f.qs.x, f.qs.y};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const unsigned w = wq[c];
      lo[2 * c] = int_pair((w & 0xFu) | ((w & 0xF00u) << 8), 0, s0);
      lo[2 * c + 1] = int_pair(((w >> 16) & 0xFu) | ((w >> 8) & 0xF0000u), 0, s0);
      hi[2 * c] = int_pair(((w >> 4) & 0xFu) | ((w << 4) & 0xF0000u), 0, s1);
      hi[2 * c + 1] = int_pair(((w >> 20) & 0xFu) | ((w >> 12) & 0xF0000u), 0, s1);
    }
    *reinterpret_cast<uint4*>(&sm.w[r][8 * q]) = uint4{lo[0], lo[1], lo[2], lo[3]};
    *reinterpret_cast<uint4*>(&sm.w[r][32 + 8 * q]) = uint4{hi[0], hi[1], hi[2], hi[3]};
    if (q == 0) {
      const float d = q4k_detail::fp16_bits(f.h.x & 0xFFFFu);
      const float dmin = q4k_detail::fp16_bits(f.h.x >> 16);
      *reinterpret_cast<float4*>(sm.sc[r]) = float4{d, dmin * sc.m0, dmin * sc.m1, 0.0f};
    }
    if (tid < kBM)
      *reinterpret_cast<float2*>(sm.xs[tid]) = f.xs;
  } else {
    const unsigned u = ss & 1, quad = 2 * u + (q >> 1);
    const unsigned lw[4] = {f.ql.x, f.ql.y, f.ql.z, f.ql.w},
                   hw[4] = {f.qh.x, f.qh.y, f.qh.z, f.qh.w};
    const half2_t one{1, 1};
    unsigned out[8];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const unsigned v =
          ((lw[c] >> (4 * u)) & 0x0F0F0F0Fu) | (((hw[c] >> (2 * quad)) & 0x03030303u) << 4);
      out[2 * c] = int_pair((v & 0xFFu) | ((v & 0xFF00u) << 8), 32, one);
      out[2 * c + 1] = int_pair(((v >> 16) & 0xFFu) | ((v >> 8) & 0xFF0000u), 32, one);
    }
    *reinterpret_cast<uint4*>(&sm.w[r][16 * q]) = uint4{out[0], out[1], out[2], out[3]};
    *reinterpret_cast<uint4*>(&sm.w[r][16 * q + 8]) = uint4{out[4], out[5], out[6], out[7]};
    sm.sc[r][q] = q4k_detail::fp16_bits(f.d) * static_cast<float>(f.sc);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tid + kThreads * i, tok = idx >> 3, c = idx & 7;
    *reinterpret_cast<uint4*>(&sm.x[tok][8 * c]) = f.x[i];
  }
}

}  // namespace qgemm_detail

// Output tiles [first, first + count) of BM tokens x BN rows, row tiles fastest.
template <QFormat F>
__device__ void qgemm_op(const QGemmParams& p, unsigned first, unsigned count) {
  using namespace qgemm_detail;
  __shared__ Smem sm;
  const int tid = static_cast<int>(threadIdx.x), tx = tid & 15, ty = tid >> 4;
  const unsigned n_tiles = (p.n_rows + kBN - 1) / kBN;
  for (unsigned tile = first + blockIdx.x; tile < first + count; tile += gridDim.x) {
    const unsigned n0 = tile % n_tiles * kBN, m0 = tile / n_tiles * kBM;
    float acc[kTM][kTN] = {};
    const unsigned n_stages = p.k / kBK;
    Fetch<F> f = fetch<F>(p, n0, m0, 0, tid);
    for (unsigned s = 0; s < n_stages; ++s) {
      store<F>(f, sm, s, tid);
      __syncthreads();
      if (s + 1 < n_stages)
        f = fetch<F>(p, n0, m0, s + 1, tid);
      float part[kTM][kTN] = {};
      // Two chunks (one Q6_K 16-group) per iteration: fully unrolled, the compiler hoists all LDS
      // reads of the stage and spills past 128 VGPRs.
#pragma unroll 2
      for (int kk = 0; kk < kBK / 8; ++kk) {
        uint4 a[kTM], b[kTN];
#pragma unroll
        for (int i = 0; i < kTM; ++i)
          a[i] = *reinterpret_cast<const uint4*>(&sm.x[ty + 16 * i][8 * kk]);
#pragma unroll
        for (int j = 0; j < kTN; ++j)
          b[j] = *reinterpret_cast<const uint4*>(&sm.w[tx + 16 * j][8 * kk]);
#pragma unroll
        for (int i = 0; i < kTM; ++i)
#pragma unroll
          for (int j = 0; j < kTN; ++j) {
            float v = part[i][j];
            v = __builtin_amdgcn_fdot2(h2(a[i].x), h2(b[j].x), v, false);
            v = __builtin_amdgcn_fdot2(h2(a[i].y), h2(b[j].y), v, false);
            v = __builtin_amdgcn_fdot2(h2(a[i].z), h2(b[j].z), v, false);
            v = __builtin_amdgcn_fdot2(h2(a[i].w), h2(b[j].w), v, false);
            part[i][j] = v;
          }
        if constexpr (F == QFormat::Q6K) {
          if (kk & 1) {
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
              const float scale = sm.sc[tx + 16 * j][kk >> 1];
#pragma unroll
              for (int i = 0; i < kTM; ++i)
                acc[i][j] += scale * part[i][j], part[i][j] = 0;
            }
          }
        }
      }
      if constexpr (F == QFormat::Q4K) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
          const float4 sc = *reinterpret_cast<const float4*>(sm.sc[tx + 16 * j]);
#pragma unroll
          for (int i = 0; i < kTM; ++i)
            acc[i][j] +=
                sc.x * part[i][j] - sc.y * sm.xs[ty + 16 * i][0] - sc.z * sm.xs[ty + 16 * i][1];
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
      const unsigned t = m0 + ty + 16 * i;
#pragma unroll
      for (int j = 0; j < kTN; ++j) {
        const unsigned n = n0 + tx + 16 * j;
        if (t < p.n_tok && n < p.n_rows)
          p.y[std::size_t{t} * p.ldy + n] = acc[i][j];
      }
    }
  }
}

__host__ __device__ inline unsigned qgemm_tiles(unsigned n_rows, unsigned n_tok) {
  using namespace qgemm_detail;
  return (n_rows + kBN - 1) / kBN * ((n_tok + kBM - 1) / kBM);
}

template <QFormat F>
__global__ void __launch_bounds__(qgemm_detail::kThreads, 2) qgemm_kernel(QGemmParams p) {
  qgemm_op<F>(p, 0, qgemm_tiles(p.n_rows, p.n_tok));
}

}  // namespace miso::kernels
