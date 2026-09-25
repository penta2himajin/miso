#pragma once

// One decode step of Gated DeltaNet (qwen3_5_moe linear attention), after the in-projections:
// causal conv1d + SiLU, q/k L2 norm, gated delta rule on the FP32 state, gated RMSNorm.
//
// Per V head h (one workgroup), with K head kh = h % 16 (GGUF stores V heads in tiled order):
//   q, k, v  = SiLU(conv(qkv)) restricted to head kh (q, k) and h (v)
//   q = q / (|q| sqrt(128)), k = k / |k|           (|x| = sqrt(sum x^2 + 1e-6))
//   beta = sigmoid(b[h]), decay = exp(ssm_a[h] * softplus(a[h] + dt_bias[h]))
//   S = decay S;  delta = (v - S^T k) beta;  S += k delta^T;  o = S^T q
//   out = o * rsqrt(mean(o^2) + eps) * norm_w * SiLU(z)
//
// The conv state is double-buffered: V heads h and h + 16 both read head kh's q/k history, so the
// step reads `conv_in` and writes `conv_out`. Head h < 16 writes the q/k channels of head h.

#include <hip/hip_runtime.h>

#include "wave.hpp"

namespace miso::kernels {

struct DeltaNetStepParams {
  const float* qkv;      // [8192] in-projection output: q[2048] k[2048] v[4096]
  const float* z;        // [4096] gate projection
  const float* a;        // [32]
  const float* b;        // [32]
  const float* conv_w;   // [8192][4], tap 3 multiplies the current input
  const float* conv_in;  // [8192][3] previous inputs, oldest first
  float* conv_out;       // [8192][3]
  const float* ssm_a;    // [32] = -exp(A_log)
  const float* dt_bias;  // [32]
  const float* norm_w;   // [128]
  float* state;          // [32][128 (k)][128 (v)]
  float* out;            // [4096]
  float eps;
};

namespace deltanet_detail {

constexpr int kKHeads = 16, kVHeads = 32, kDim = 128;

__device__ inline float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ inline float softplus(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ inline float conv_step(const DeltaNetStepParams& p, unsigned c, bool own) {
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

}  // namespace deltanet_detail

// V heads [first, first + count), one 256-thread workgroup per head (grid-stride).
template <int kBlock>
__device__ void deltanet_step_op(const DeltaNetStepParams& p, unsigned first, unsigned count) {
  using namespace deltanet_detail;
  static_assert(kBlock == 2 * kDim, "thread t owns v column t % 128 and k half t / 128");
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

    // L2 norms: waves 0-1 hold q, waves 2-3 hold k.
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

    // Gated RMSNorm over the head's 128 outputs (waves 0-1).
    const float oh = t < kDim ? part[0][t] + part[1][t] : 0.0f;
    const float os = wave_sum(oh * oh);
    if (t % kWave == kWave - 1)
      red[wave] = os;
    __syncthreads();
    if (t < kDim) {
      const float r = rsqrtf((red[0] + red[1]) / kDim + p.eps);
      p.out[h * kDim + t] = oh * r * p.norm_w[t] * silu(p.z[h * kDim + t]);
    }
    __syncthreads();
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) deltanet_step_kernel(DeltaNetStepParams p) {
  deltanet_step_op<kBlock>(p, 0, deltanet_detail::kVHeads);
}

// The same recurrence over n consecutive tokens in one pass (prefill): each workgroup keeps its
// head's state in registers and the conv history of its channels in a register window, so only
// the per-token inputs and outputs touch memory. Per token the arithmetic is that of
// deltanet_step_op, so the result is bit-identical to n steps.
struct DeltaNetSeqParams {
  const float* qkv;  // [n][8192]
  const float* z;    // [n][zab_stride], first 4096 rows
  const float* a;    // [n][zab_stride], rows 4096..4127
  const float* b;    // [n][zab_stride], rows 4128..4159
  const float* conv_w;
  const float* conv_in;  // [8192][3] before the first token
  float* conv_out;       // [8192][3] after the last token
  const float* ssm_a;
  const float* dt_bias;
  const float* norm_w;
  float* state;
  float* out;  // [n][4096]
  unsigned n;
  float eps;
  unsigned zab_stride;  // 4160 when z, a and b were concatenated into one GEMV output
};

template <int kBlock>
__device__ void deltanet_seq_op(const DeltaNetSeqParams& p, unsigned first, unsigned count) {
  using namespace deltanet_detail;
  static_assert(kBlock == 2 * kDim, "thread t owns v column t % 128 and k half t / 128");
  constexpr unsigned kQkv = 2 * kKHeads * kDim + kVHeads * kDim, kO = kVHeads * kDim;
  __shared__ float q[kDim], k[kDim], v[kDim], red[kBlock / kWave], part[2][kDim];
  const unsigned t = threadIdx.x, vi = t % kDim, half = t / kDim, wave = t / kWave;

  for (unsigned h = first + blockIdx.x; h < first + count; h += gridDim.x) {
    const unsigned kh = h % kKHeads;
    const bool own_qk = h < kKHeads;
    // Channel c0: q (t < 128) or k; channel c1: v (t < 128 only).
    const unsigned c0 = t < kDim ? kh * kDim + t : kKHeads * kDim + kh * kDim + vi;
    const unsigned c1 = 2 * kKHeads * kDim + h * kDim + vi;
    float w0[4], w1[4], s0[3], s1[3];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      w0[i] = p.conv_w[4 * c0 + i], w1[i] = p.conv_w[4 * c1 + i];
#pragma unroll
    for (int i = 0; i < 3; ++i)
      s0[i] = p.conv_in[3 * c0 + i], s1[i] = p.conv_in[3 * c1 + i];
    float* S = p.state + std::size_t{h} * kDim * kDim;
    float s[kDim / 2];
#pragma unroll
    for (int j = 0; j < kDim / 2; ++j)
      s[j] = S[(half * (kDim / 2) + j) * kDim + vi];

    for (unsigned tok = 0; tok < p.n; ++tok) {
      const float* x = p.qkv + std::size_t{tok} * kQkv;
      const float x0 = x[c0];
      const float y0 = silu(w0[0] * s0[0] + w0[1] * s0[1] + w0[2] * s0[2] + w0[3] * x0);
      s0[0] = s0[1], s0[1] = s0[2], s0[2] = x0;
      if (t < kDim) {
        const float x1 = x[c1];
        const float y1 = silu(w1[0] * s1[0] + w1[1] * s1[1] + w1[2] * s1[2] + w1[3] * x1);
        s1[0] = s1[1], s1[1] = s1[2], s1[2] = x1;
        q[t] = y0, v[t] = y1;
      } else {
        k[vi] = y0;
      }
      __syncthreads();

      const float xn = t < kDim ? q[t] : k[vi];
      const float ss = wave_sum(xn * xn);
      if (t % kWave == kWave - 1)
        red[wave] = ss;
      __syncthreads();
      if (t < kDim) {
        q[t] = xn * rsqrtf(red[0] + red[1] + 1e-6f) * rsqrtf(static_cast<float>(kDim));
      } else {
        k[vi] = xn * rsqrtf(red[2] + red[3] + 1e-6f);
      }
      __syncthreads();

      const float beta = 1.0f / (1.0f + __expf(-p.b[tok * p.zab_stride + h]));
      const float decay = __expf(p.ssm_a[h] * softplus(p.a[tok * p.zab_stride + h] + p.dt_bias[h]));
      float kv = 0;
#pragma unroll
      for (int j = 0; j < kDim / 2; ++j) {
        s[j] = s[j] * decay;
        kv += s[j] * k[half * (kDim / 2) + j];
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
        p.out[std::size_t{tok} * kO + h * kDim + t] =
            oh * r * p.norm_w[t] * silu(p.z[std::size_t{tok} * p.zab_stride + h * kDim + t]);
      }
      __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < kDim / 2; ++j)
      S[(half * (kDim / 2) + j) * kDim + vi] = s[j];
    if (own_qk) {
#pragma unroll
      for (int i = 0; i < 3; ++i)
        p.conv_out[3 * c0 + i] = s0[i];
    }
    if (t < kDim) {
#pragma unroll
      for (int i = 0; i < 3; ++i)
        p.conv_out[3 * c1 + i] = s1[i];
    }
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) deltanet_seq_kernel(DeltaNetSeqParams p) {
  deltanet_seq_op<kBlock>(p, 0, deltanet_detail::kVHeads);
}

}  // namespace miso::kernels
