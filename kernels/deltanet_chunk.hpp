#pragma once

// Prefill Gated DeltaNet as a batched conv plus the chunked gated delta rule (chunk 64).
// The recurrence per V head is the one in deltanet_step.hpp:
//   S = α S;  δ = β (v - S^T k);  S += k δ^T;  o = S^T q
// Over a chunk, with g[t] = sum_{i<=t} log α_i and γ = exp(g):
//   (I + L) δ = β (v - γ (S^T k)),  L[t,j] = β_t γ_t/γ_j (k_j·k_t) for j < t
//   o_t = γ_t (S^T q_t) + sum_{j<=t} (γ_t/γ_j) (k_j·q_t) δ_j
//   S    = γ_last S + sum_j (γ_last/γ_j) k_j δ_j^T
// Conv, SiLU and the q/k L2 norms match deltanet_seq_op, so the state and outputs agree with the
// decode recurrence up to this reassociation.
// On gfx906 this kernel is 256 VGPR with 860 B of scratch (214 spills). Wired into prefill it
// dropped pp512 from 1397.5 to 387.8 tok/s, so deltanet_prefill keeps deltanet_seq_kernel.

#include <hip/hip_runtime.h>

#include "deltanet_step.hpp"
#include "wave.hpp"

namespace miso::kernels {

constexpr unsigned kDeltaChunk = 64;

struct DeltaNetConvParams {
  const float* qkv;  // [n][8192]
  const float* a;    // [n][32]
  const float* b;    // [n][32]
  const float* conv_w;
  const float* conv_in;
  float* conv_out;
  const float* ssm_a;
  const float* dt_bias;
  float* q;      // [n][16][128] normalised
  float* k;      // [n][16][128]
  float* v;      // [n][32][128]
  float* beta;   // [n][32]
  float* log_a;  // [n][32] log α
  unsigned n;
};

// Channels [first, first + count) of 8192. One thread walks every token of one channel.
__device__ inline void deltanet_conv_op(const DeltaNetConvParams& p, unsigned first,
                                        unsigned count) {
  using namespace deltanet_detail;
  for (unsigned c = first + blockIdx.x * blockDim.x + threadIdx.x; c < first + count;
       c += gridDim.x * blockDim.x) {
    float w0 = p.conv_w[4 * c], w1 = p.conv_w[4 * c + 1], w2 = p.conv_w[4 * c + 2],
          w3 = p.conv_w[4 * c + 3];
    float s0 = p.conv_in[3 * c], s1 = p.conv_in[3 * c + 1], s2 = p.conv_in[3 * c + 2];
    for (unsigned tok = 0; tok < p.n; ++tok) {
      const float x = p.qkv[std::size_t{tok} * (kKHeads * kDim * 2 + kVHeads * kDim) + c];
      const float y = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
      s0 = s1, s1 = s2, s2 = x;
      // q, k, v are normalised by a later kernel; stash the SiLU value in v's buffer for v
      // channels and in a side slot. q/k channels are written raw into q/k and normalised in place.
      if (c < kKHeads * kDim) {
        p.q[std::size_t{tok} * kKHeads * kDim + c] = y;
      } else if (c < 2 * kKHeads * kDim) {
        p.k[std::size_t{tok} * kKHeads * kDim + (c - kKHeads * kDim)] = y;
      } else {
        p.v[std::size_t{tok} * kVHeads * kDim + (c - 2 * kKHeads * kDim)] = y;
      }
    }
    p.conv_out[3 * c] = s0;
    p.conv_out[3 * c + 1] = s1;
    p.conv_out[3 * c + 2] = s2;
  }
}

__global__ void __launch_bounds__(256) deltanet_conv_kernel(DeltaNetConvParams p) {
  deltanet_conv_op(p, 0,
                   deltanet_detail::kKHeads * deltanet_detail::kDim * 2 +
                       deltanet_detail::kVHeads * deltanet_detail::kDim);
}

// One K head (q and k) or the beta / log α of one V head, for every token.
// Items: [0, 16) q heads, [16, 32) k heads, [32, 64) V-head beta and log α. Grid-stride.
__device__ inline void deltanet_norm_op(const DeltaNetConvParams& p, unsigned first,
                                        unsigned count) {
  using namespace deltanet_detail;
  __shared__ float red[256 / kWave];
  const unsigned t = threadIdx.x;
  for (unsigned item = first + blockIdx.x; item < first + count; item += gridDim.x) {
    for (unsigned tok = 0; tok < p.n; ++tok) {
      if (item < 2 * kKHeads) {
        float* row =
            (item < kKHeads ? p.q : p.k) + (std::size_t{tok} * kKHeads + item % kKHeads) * kDim;
        const float x = t < kDim ? row[t] : 0.0f;
        const float ss = wave_sum(x * x);
        if (t % kWave == kWave - 1)
          red[t / kWave] = ss;
        __syncthreads();
        if (t < kDim) {
          float r = rsqrtf(red[0] + red[1] + 1e-6f);
          if (item < kKHeads)
            r *= rsqrtf(static_cast<float>(kDim));
          row[t] = x * r;
        }
        __syncthreads();
      } else if (t == 0) {
        const unsigned h = item - 2 * kKHeads;
        const float z = p.a[tok * kVHeads + h] + p.dt_bias[h];
        p.beta[tok * kVHeads + h] = 1.0f / (1.0f + __expf(-p.b[tok * kVHeads + h]));
        p.log_a[tok * kVHeads + h] = p.ssm_a[h] * (z > 20.0f ? z : log1pf(__expf(z)));
      }
    }
  }
}

__global__ void __launch_bounds__(256) deltanet_norm_kernel(DeltaNetConvParams p) {
  deltanet_norm_op(p, 0, deltanet_detail::kKHeads * 2 + deltanet_detail::kVHeads);
}

struct DeltaNetChunkParams {
  const float* q;      // [n][16][128]
  const float* k;      // [n][16][128]
  const float* v;      // [n][32][128]
  const float* beta;   // [n][32]
  const float* log_a;  // [n][32]
  const float* z;      // [n][4096]
  const float* norm_w;
  float* state;  // [32][128][128]
  float* out;    // [n][4096]
  float* sk;     // [32][64][128] scratch
  float* sq;     // [32][64][128]
  unsigned n;
  float eps;
};

// V heads [first, first + count). State stays in registers across chunks of 64.
template <int kBlock>
__device__ void deltanet_chunk_op(const DeltaNetChunkParams& p, unsigned first, unsigned count) {
  using namespace deltanet_detail;
  static_assert(kBlock == 2 * kDim);
  constexpr unsigned kC = kDeltaChunk;
  __shared__ float kk[kC * kC];
  __shared__ float L[kC * kC];
  __shared__ float part[2][16][kDim];
  __shared__ float red[kBlock / kWave];
  __shared__ float g[kC], beta[kC], scale[kC];
  const unsigned t = threadIdx.x, vi = t % kDim, half = t / kDim;

  for (unsigned h = first + blockIdx.x; h < first + count; h += gridDim.x) {
    const unsigned kh = h % kKHeads;
    float* S = p.state + std::size_t{h} * kDim * kDim;
    float s[kDim / 2];
#pragma unroll
    for (int j = 0; j < kDim / 2; ++j)
      s[j] = S[(half * (kDim / 2) + j) * kDim + vi];

    for (unsigned t0 = 0; t0 < p.n; t0 += kC) {
      const unsigned C = min(kC, p.n - t0);
      const float* krow = p.k + (std::size_t{t0} * kKHeads + kh) * kDim;
      const float* qrow = p.q + (std::size_t{t0} * kKHeads + kh) * kDim;
      const unsigned kk_stride = kKHeads * kDim;

      for (unsigned e = t; e < C * C; e += kBlock) {
        const unsigned i = e / C, j = e % C;
        const float* a = krow + i * kk_stride;
        const float* b = krow + j * kk_stride;
        float d = 0;
        for (int c = 0; c < kDim; ++c)
          d += a[c] * b[c];
        kk[i * kC + j] = d;
      }
      if (t < C) {
        float acc = 0;
        for (unsigned i = 0; i <= t; ++i)
          acc += p.log_a[(t0 + i) * kVHeads + h];
        g[t] = acc;
        beta[t] = p.beta[(t0 + t) * kVHeads + h];
      }
      __syncthreads();

      auto dots = [&](const float* rows, float* dst) {
        for (unsigned base = 0; base < C; base += 16) {
          const unsigned m = min(16u, C - base);
          for (unsigned u = 0; u < m; ++u) {
            float partial = 0;
            for (int j = 0; j < kDim / 2; ++j) {
              const unsigned kk_i = half * (kDim / 2) + j;
              partial += s[j] * rows[(base + u) * kk_stride + kk_i];
            }
            part[half][u][vi] = partial;
          }
          __syncthreads();
          if (half == 0) {
#pragma unroll
            for (unsigned u = 0; u < 16; ++u)
              if (u < m)
                dst[(base + u) * kDim + vi] = part[0][u][vi] + part[1][u][vi];
          }
          __syncthreads();
        }
      };
      dots(krow, p.sk + std::size_t{h} * kC * kDim);
      dots(qrow, p.sq + std::size_t{h} * kC * kDim);

      for (unsigned e = t; e < C * C; e += kBlock) {
        const unsigned i = e / C, j = e % C;
        L[i * kC + j] = (j < i) ? beta[i] * __expf(g[i] - g[j]) * kk[j * kC + i] : 0.0f;
      }
      for (unsigned e = t; e < C * C; e += kBlock) {
        const unsigned i = e / C, j = e % C;
        const float* a = qrow + i * kk_stride;
        const float* b = krow + j * kk_stride;
        float d = 0;
        for (int c = 0; c < kDim; ++c)
          d += a[c] * b[c];
        kk[i * kC + j] = d;  // q_i · k_j
      }
      if (t < C)
        scale[t] = __expf(g[t]);
      __syncthreads();

      float* dlt = p.sk + std::size_t{h} * kC * kDim;  // sk[i] is consumed as δ[i] is stored
      const float* sq = p.sq + std::size_t{h} * kC * kDim;
      const float* vv = p.v + (std::size_t{t0} * kVHeads + h) * kDim;
#pragma unroll 1
      for (unsigned i = 0; i < C; ++i) {
        float acc = beta[i] * (vv[i * kVHeads * kDim + vi] - scale[i] * dlt[i * kDim + vi]);
        for (unsigned j = 0; j < i; ++j)
          acc -= L[i * kC + j] * dlt[j * kDim + vi];
        if (half == 0)
          dlt[i * kDim + vi] = acc;
      }
      __syncthreads();

#pragma unroll 1
      for (unsigned i = 0; i < C; ++i) {
        float o = scale[i] * sq[i * kDim + vi];
        for (unsigned j = 0; j <= i; ++j)
          o += __expf(g[i] - g[j]) * kk[i * kC + j] * dlt[j * kDim + vi];
        const float oh = t < kDim ? o : 0.0f;
        const float os = wave_sum(oh * oh);
        if (t % kWave == kWave - 1)
          red[t / kWave] = os;
        __syncthreads();
        if (t < kDim) {
          const float r = rsqrtf((red[0] + red[1]) / kDim + p.eps);
          const float gate = p.z[(std::size_t{t0 + i} * kVHeads + h) * kDim + t];
          p.out[(std::size_t{t0 + i} * kVHeads + h) * kDim + t] =
              oh * r * p.norm_w[t] * (gate / (1.0f + __expf(-gate)));
        }
        __syncthreads();
      }

      const float g_last = __expf(g[C - 1]);
      for (int j = 0; j < kDim / 2; ++j) {
        const unsigned kk_i = half * (kDim / 2) + j;
        float add = 0;
#pragma unroll 1
        for (unsigned i = 0; i < C; ++i)
          add += __expf(g[C - 1] - g[i]) * krow[i * kk_stride + kk_i] * dlt[i * kDim + vi];
        s[j] = g_last * s[j] + add;
      }
      __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < kDim / 2; ++j)
      S[(half * (kDim / 2) + j) * kDim + vi] = s[j];
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) deltanet_chunk_kernel(DeltaNetChunkParams p) {
  deltanet_chunk_op<kBlock>(p, 0, deltanet_detail::kVHeads);
}

}  // namespace miso::kernels
