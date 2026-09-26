#pragma once

// Gated full attention, prefill path (ADR_001 D3 prefill family), for n consecutive tokens at
// positions pos0 .. pos0 + n - 1:
//   attn_prep_batch: the decode per-head prep (norms, RoPE, KV append) for all n x 18 heads.
//   attn_prefill:    causal attention; a workgroup takes one KV head and kQBlock consecutive
//                    tokens (8 query heads each), stages each 32-position K sub-chunk in LDS once
//                    for all of them and keeps an online softmax per (token, head) row. Per row
//                    the arithmetic follows attn_split_op; output is gated as in attn_combine.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "attention_decode.hpp"
#include "wave.hpp"

namespace miso::kernels {

// Items [first, first + count) of 18 n (token-major).
template <int kBlock>
__device__ void attn_prep_batch_op(const AttnPrepBatchParams& p, unsigned first, unsigned count) {
  using namespace attn;
  constexpr unsigned kHeads = kQHeads + kKvHeads;
  for (unsigned item = first + blockIdx.x; item < first + count; item += gridDim.x) {
    const unsigned t = item / kHeads, head = item % kHeads;
    AttnPrepParams q = p.tok0;
    q.qg += std::size_t{t} * p.proj_stride;
    q.k += std::size_t{t} * p.proj_stride;
    q.v += std::size_t{t} * p.proj_stride;
    q.q += std::size_t{t} * kQHeads * kDim;
    q.pos += t;
    attn_prep_head<kBlock>(q, head);
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) attn_prep_batch_kernel(AttnPrepBatchParams p) {
  attn_prep_batch_op<kBlock>(p, 0, p.n * (attn::kQHeads + attn::kKvHeads));
}

struct AttnPrefillParams {
  const float* q;               // [n][16][256] normalised, rotated (unscaled)
  const float* qg;              // token-major; gate at tok * qg_stride + head * 512 + 256
  const attn::half_t* k_cache;  // [2][max_ctx][256], positions [0, pos0 + n) valid
  const attn::half_t* v_cache;
  float* core;  // [n][16][256]
  unsigned pos0, n, max_ctx;
  unsigned qg_stride;  // 8192 packed, or 9216 when q/k/v share a proj buffer
};

constexpr unsigned kAttnQBlock = 4;

__host__ __device__ inline unsigned attn_prefill_items(unsigned n) {
  return attn::kKvHeads * ((n + kAttnQBlock - 1) / kAttnQBlock);
}

// Items [first, first + count): item = block * 2 + KV head.
template <int kBlock>
__device__ void attn_prefill_op(const AttnPrefillParams& p, unsigned first, unsigned count) {
  using namespace attn;
  constexpr int kGroup = kQHeads / kKvHeads, kSub = kAttnSubChunk, kRow = kDim / 2 + 1;
  constexpr int kQB = kAttnQBlock;
  static_assert(kBlock == kDim && kBlock == kGroup * kSub);
  typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
  __shared__ float qs[kQB][kGroup][kDim], pr[kQB][kGroup][kSub];
  __shared__ float m_s[kQB][kGroup], l_s[kQB][kGroup], corr_s[kQB][kGroup];
  __shared__ half2_t ks[kSub][kRow];
  const unsigned t = threadIdx.x, hl = t / kSub, jl = t % kSub;
  const float scale = rsqrtf(static_cast<float>(kDim));

  for (unsigned item = first + blockIdx.x; item < first + count; item += gridDim.x) {
    const unsigned kvh = item % kKvHeads, t0 = item / kKvHeads * kQB;
    const unsigned nq = min(unsigned{kQB}, p.n - t0);
    const half_t* K = p.k_cache + std::size_t{kvh} * p.max_ctx * kDim;
    const half_t* V = p.v_cache + std::size_t{kvh} * p.max_ctx * kDim;
    for (int r = 0; r < kQB; ++r)
      for (int h = 0; h < kGroup; ++h)
        qs[r][h][t] =
            r < static_cast<int>(nq)
                ? p.q[(std::size_t{t0 + r} * kQHeads + kvh * kGroup + h) * kDim + t] * scale
                : 0.0f;
    if (t < kQB * kGroup)
      m_s[t / kGroup][t % kGroup] = -__builtin_inff(), l_s[t / kGroup][t % kGroup] = 0;
    float acc[kQB][kGroup] = {};
    const unsigned len = p.pos0 + t0 + nq;  // keys of the block's last token
    __syncthreads();

    for (unsigned j0 = 0; j0 < len; j0 += kSub) {
      const unsigned n = min(unsigned{kSub}, len - j0);
      {
        const unsigned row = t / 8, col = (t % 8) * 16;
        if (row < n) {
          const uint4* src =
              reinterpret_cast<const uint4*>(K + std::size_t{j0 + row} * kDim) + (t % 8) * 4;
          for (int c = 0; c < 4; ++c) {
            const uint4 v = src[c];
            const unsigned w[4] = {v.x, v.y, v.z, v.w};
            for (int e = 0; e < 4; ++e)
              ks[row][col + 4 * c + e] = __builtin_bit_cast(half2_t, w[e]);
          }
        }
      }
      __syncthreads();
      for (int r = 0; r < kQB; ++r) {
        // Token t0 + r sees positions [0, pos0 + t0 + r].
        const unsigned lim = p.pos0 + t0 + r + 1;
        float s = -__builtin_inff();
        if (r < static_cast<int>(nq) && jl < n && j0 + jl < lim) {
          s = 0;
          for (int e = 0; e < kDim / 2; ++e) {
            const half2_t kk = ks[jl][e];
            s += qs[r][hl][2 * e] * static_cast<float>(kk.x) +
                 qs[r][hl][2 * e + 1] * static_cast<float>(kk.y);
          }
        }
        float mx = s;
        for (int off = kSub / 2; off > 0; off >>= 1)
          mx = fmaxf(mx, __shfl_xor(mx, off, kSub));
        const float m_new = fmaxf(m_s[r][hl], mx);
        const float pj = s == -__builtin_inff() ? 0.0f : __expf(s - m_new);
        float sum = pj;
        for (int off = kSub / 2; off > 0; off >>= 1)
          sum += __shfl_xor(sum, off, kSub);
        pr[r][hl][jl] = pj;
        __syncthreads();
        if (jl == 0) {
          const float c = m_new == -__builtin_inff() ? 1.0f : __expf(m_s[r][hl] - m_new);
          corr_s[r][hl] = c;
          l_s[r][hl] = l_s[r][hl] * c + sum;
          m_s[r][hl] = m_new;
        }
      }
      __syncthreads();
      for (int r = 0; r < kQB; ++r)
        for (int h = 0; h < kGroup; ++h)
          acc[r][h] *= corr_s[r][h];
      for (unsigned j = 0; j < n; ++j) {
        const float v = static_cast<float>(V[std::size_t{j0 + j} * kDim + t]);
        for (int r = 0; r < kQB; ++r)
          for (int h = 0; h < kGroup; ++h)
            acc[r][h] += pr[r][h][j] * v;
      }
      __syncthreads();
    }
    for (int r = 0; r < static_cast<int>(nq); ++r) {
      for (int h = 0; h < kGroup; ++h) {
        const std::size_t head = std::size_t{t0 + r} * kQHeads + kvh * kGroup + h;
        const float gate =
            p.qg[std::size_t{t0 + r} * p.qg_stride + (kvh * kGroup + h) * 2 * kDim + kDim + t];
        p.core[head * kDim + t] = acc[r][h] / l_s[r][h] / (1.0f + __expf(-gate));
      }
    }
    __syncthreads();
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) attn_prefill_kernel(AttnPrefillParams p) {
  attn_prefill_op<kBlock>(p, 0, attn_prefill_items(p.n));
}

}  // namespace miso::kernels
