#pragma once

// Gated full attention (qwen3_5_moe), decode path: 16 query heads, 2 KV heads, head_dim 256,
// partial NEOX RoPE on the first 64 dims, FP16 KV cache (ADR_002 D9).

#include <hip/hip_runtime.h>

#include <cstdint>

#include "wave.hpp"

namespace miso::kernels {

namespace attn {
constexpr int kQHeads = 16, kKvHeads = 2, kDim = 256, kRopeDim = 64;
typedef _Float16 half_t;
}  // namespace attn

struct AttnPrepParams {
  const float* qg;        // [16][512]: per head 256 query then 256 gate (q projection output)
  const float* k;         // [2][256]
  const float* v;         // [2][256]
  const float* q_norm;    // [256], GGUF stores 1 + w
  const float* k_norm;    // [256]
  const float* inv_freq;  // [32]
  attn::half_t* k_cache;  // [2][max_ctx][256]
  attn::half_t* v_cache;  // [2][max_ctx][256]
  float* q;               // [16][256] normalised, rotated queries
  unsigned pos, max_ctx;
  float eps;
};

// Prefill: tok0 pointers address token 0; each later token is `t * proj_stride` floats further in
// qg/k/v (9216 when q/k/v share one buffer: [q 8192 | k 512 | v 512]).
struct AttnPrepBatchParams {
  AttnPrepParams tok0;
  unsigned n;
  unsigned proj_stride;
};

// One head of 18 (0..15 query heads, 16..17 KV heads) on a 256-thread workgroup: per-head RMSNorm,
// RoPE at `pos`, then write q or append k / v to the cache.
template <int kBlock>
__device__ void attn_prep_head(const AttnPrepParams& p, unsigned head) {
  using namespace attn;
  static_assert(kBlock == kDim);
  __shared__ float xs[kDim], red[kBlock / kWave];
  const unsigned d = threadIdx.x;
  const bool is_q = head < kQHeads;
  const unsigned kvh = head - kQHeads;
  float x = is_q ? p.qg[head * 2 * kDim + d] : p.k[kvh * kDim + d];
  const float r = rsqrtf(block_sum<kBlock>(x * x, red) / kDim + p.eps);
  x = x * r * (is_q ? p.q_norm[d] : p.k_norm[d]);
  xs[d] = x;
  __syncthreads();
  if (d < kRopeDim) {
    constexpr unsigned kHalf = kRopeDim / 2;
    const unsigned i = d % kHalf;
    float s, c;
    sincosf(static_cast<float>(p.pos) * p.inv_freq[i], &s, &c);
    x = d < kHalf ? xs[i] * c - xs[i + kHalf] * s : xs[i + kHalf] * c + xs[i] * s;
  }
  if (is_q) {
    p.q[head * kDim + d] = x;
  } else {
    const std::size_t at = (std::size_t{kvh} * p.max_ctx + p.pos) * kDim + d;
    p.k_cache[at] = static_cast<half_t>(x);
    p.v_cache[at] = static_cast<half_t>(p.v[kvh * kDim + d]);
  }
  __syncthreads();
}

// Heads [first, first + count) of 18, one workgroup per head (grid-stride).
template <int kBlock>
__device__ void attn_prep_op(const AttnPrepParams& p, unsigned first, unsigned count) {
  for (unsigned head = first + blockIdx.x; head < first + count; head += gridDim.x)
    attn_prep_head<kBlock>(p, head);
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) attn_prep_kernel(AttnPrepParams p) {
  attn_prep_op<kBlock>(p, 0, attn::kQHeads + attn::kKvHeads);
}

// Split-K flash decoding. The positions [0, len) of each KV head are divided among n_splits
// workgroups; each handles the 8 query heads sharing that KV head, so every K/V row is read once.
// A workgroup walks its span in sub-chunks of 32 positions with an online softmax and writes a
// partial (max, sum, unnormalised output) per query head; attn_combine merges the partials.

constexpr unsigned kAttnSubChunk = 32;

// Number of splits per KV head: at most max_splits, each split non-empty.
__host__ __device__ constexpr unsigned attn_n_splits(unsigned len, unsigned max_splits) {
  const unsigned subs = (len + kAttnSubChunk - 1) / kAttnSubChunk;
  const unsigned per = (subs + max_splits - 1) / max_splits;
  return (subs + per - 1) / per;
}

struct AttnSplitParams {
  const float* q;               // [16][256] (unscaled)
  const attn::half_t* k_cache;  // [2][max_ctx][256]
  const attn::half_t* v_cache;
  float* part_m;  // [2][n_splits][8] running max of scaled scores
  float* part_l;  // [2][n_splits][8] sum of exp(score - max)
  float* part_o;  // [2][n_splits][8][256] sum of exp(score - max) V
  unsigned len, max_ctx, n_splits;
};

template <int kBlock>
__device__ void attn_split_op(const AttnSplitParams& p, unsigned first, unsigned count) {
  using namespace attn;
  constexpr int kGroup = kQHeads / kKvHeads, kSub = kAttnSubChunk, kRow = kDim / 2 + 1;
  static_assert(kBlock == kDim && kBlock == kGroup * kSub);
  typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
  __shared__ float qs[kGroup][kDim], pr[kGroup][kSub], m_s[kGroup], l_s[kGroup], corr_s[kGroup];
  __shared__ half2_t ks[kSub][kRow];  // padded rows: consecutive positions hit different banks
  const unsigned t = threadIdx.x, hl = t / kSub, jl = t % kSub;
  const float scale = rsqrtf(static_cast<float>(kDim));
  const unsigned subs = (p.len + kSub - 1) / kSub;
  const unsigned per = (subs + p.n_splits - 1) / p.n_splits;

  for (unsigned item = first + blockIdx.y * gridDim.x + blockIdx.x; item < first + count;
       item += gridDim.x * gridDim.y) {
    const unsigned kvh = item / p.n_splits, split = item % p.n_splits;
    const half_t* K = p.k_cache + std::size_t{kvh} * p.max_ctx * kDim;
    const half_t* V = p.v_cache + std::size_t{kvh} * p.max_ctx * kDim;
    for (int h = 0; h < kGroup; ++h)
      qs[h][t] = p.q[(kvh * kGroup + h) * kDim + t] * scale;
    if (t < kGroup)
      m_s[t] = -__builtin_inff(), l_s[t] = 0;
    float acc[kGroup] = {};
    const unsigned j_begin = split * per * kSub;
    const unsigned j_end = min(p.len, (split + 1) * per * kSub);
    __syncthreads();

    for (unsigned j0 = j_begin; j0 < j_end; j0 += kSub) {
      const unsigned n = min(kSub, j_end - j0);
      // Stage K rows [j0, j0 + n) in LDS: thread t copies 64 bytes of row t / 8.
      {
        const unsigned row = t / 8, col = (t % 8) * 16;  // in half2 units
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
      float s = -__builtin_inff();
      if (jl < n) {
        s = 0;
        for (int e = 0; e < kDim / 2; ++e) {
          const half2_t kk = ks[jl][e];
          s += qs[hl][2 * e] * static_cast<float>(kk.x) +
               qs[hl][2 * e + 1] * static_cast<float>(kk.y);
        }
      }
      // Max / sum over the 32 lanes of query head hl (half a wavefront).
      float mx = s;
      for (int off = kSub / 2; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor(mx, off, kSub));
      const float m_new = fmaxf(m_s[hl], mx);
      const float pj = jl < n ? __expf(s - m_new) : 0.0f;
      float sum = pj;
      for (int off = kSub / 2; off > 0; off >>= 1)
        sum += __shfl_xor(sum, off, kSub);
      pr[hl][jl] = pj;
      __syncthreads();
      if (jl == 0) {
        const float c = __expf(m_s[hl] - m_new);
        corr_s[hl] = c;
        l_s[hl] = l_s[hl] * c + sum;
        m_s[hl] = m_new;
      }
      __syncthreads();
      for (int h = 0; h < kGroup; ++h)
        acc[h] *= corr_s[h];
      for (unsigned j = 0; j < n; ++j) {
        const float v = static_cast<float>(V[std::size_t{j0 + j} * kDim + t]);
        for (int h = 0; h < kGroup; ++h)
          acc[h] += pr[h][j] * v;
      }
      __syncthreads();
    }
    const std::size_t base = std::size_t{kvh} * p.n_splits + split;
    if (t < kGroup)
      p.part_m[base * kGroup + t] = m_s[t], p.part_l[base * kGroup + t] = l_s[t];
    for (int h = 0; h < kGroup; ++h)
      p.part_o[(base * kGroup + h) * kDim + t] = acc[h];
    __syncthreads();
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) attn_split_kernel(AttnSplitParams p) {
  attn_split_op<kBlock>(p, 0, attn::kKvHeads * p.n_splits);
}

struct AttnCombineParams {
  const float* part_m;
  const float* part_l;
  const float* part_o;
  const float* qg;  // [16][512], for the gate half
  float* core;      // [16][256] = softmax(q K^T / 16) V * sigmoid(gate)
  unsigned n_splits;
};

// Query heads [first, first + count), one 256-thread workgroup per head.
template <int kBlock>
__device__ void attn_combine_op(const AttnCombineParams& p, unsigned first, unsigned count) {
  using namespace attn;
  constexpr int kGroup = kQHeads / kKvHeads;
  static_assert(kBlock == kDim);
  const unsigned d = threadIdx.x;
  for (unsigned h = first + blockIdx.x; h < first + count; h += gridDim.x) {
    const unsigned kvh = h / kGroup, hl = h % kGroup;
    float M = -__builtin_inff();
    for (unsigned c = 0; c < p.n_splits; ++c)
      M = fmaxf(M, p.part_m[(kvh * p.n_splits + c) * kGroup + hl]);
    float L = 0, o = 0;
    for (unsigned c = 0; c < p.n_splits; ++c) {
      const std::size_t i = (std::size_t{kvh} * p.n_splits + c) * kGroup + hl;
      const float w = __expf(p.part_m[i] - M);
      L += w * p.part_l[i];
      o += w * p.part_o[i * kDim + d];
    }
    const float gate = p.qg[h * 2 * kDim + kDim + d];
    p.core[h * kDim + d] = o / L / (1.0f + __expf(-gate));
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) attn_combine_kernel(AttnCombineParams p) {
  attn_combine_op<kBlock>(p, 0, attn::kQHeads);
}

}  // namespace miso::kernels
