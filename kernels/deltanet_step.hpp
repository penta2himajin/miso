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

}  // namespace miso::kernels
