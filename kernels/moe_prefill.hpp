#pragma once

// Sparse MoE block, prefill path (ADR_001 D3 prefill family), for a chunk of n tokens:
//   moe_router_batch: logits [n][257], bit-identical to moe_router (shared lane dot product).
//   moe_topk_batch:   per token, the decode top-8 (same function) plus slot 8 = shared expert
//                     (id 256, weight sigmoid(logits[256])). Assignment a = 9 t + k.
//   moe_group:        counting sort of the 9 n assignments by expert, cut into tasks of at most
//                     kTaskTok assignments of one expert.
//   moe_gate_up_grouped / moe_down_grouped: one lane per weight row (gate/up: a row pair), the
//                     expanded weights reused for every assignment of the task. Activations are
//                     wave-uniform, so they are read with scalar loads.
//   moe_combine:      out[t] = sum over k of the 9 weighted slot outputs, in slot order.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "moe_decode.hpp"
#include "q4k_gemv.hpp"
#include "wave.hpp"

namespace miso::kernels {

namespace moe {
constexpr int kTaskTok = 16;  // assignments per grouped-GEMM task
}  // namespace moe

struct MoeRouterBatchParams {
  const std::uint16_t* w;  // [257][2048] BF16
  const float* x;          // [n_tok][2048]
  float* logits;           // [n_tok][257]
  unsigned n_tok;
};

namespace moe_detail {
constexpr int kRouterRows = 4;  // rows per wavefront, kept in registers
constexpr int kRouterTok = 32;  // tokens per workgroup
constexpr int kRouterGroups = (moe::kExperts + 1 + kRouterRows - 1) / kRouterRows;
}  // namespace moe_detail

// Work item = (group of 4 rows per wave x 4 waves, block of kRouterTok tokens), rows fastest.
template <int kBlock>
__device__ void moe_router_batch_op(const MoeRouterBatchParams& p, unsigned first, unsigned count) {
  using namespace moe_detail;
  constexpr int kWaves = kBlock / kWave;
  constexpr unsigned kRowItems = (kRouterGroups + kWaves - 1) / kWaves;
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave;
  for (unsigned item = first + blockIdx.x; item < first + count; item += gridDim.x) {
    const unsigned g = item % kRowItems * kWaves + wave, t0 = item / kRowItems * kRouterTok;
    uint4 v[kRouterRows][4];
#pragma unroll
    for (int i = 0; i < kRouterRows; ++i) {
      const unsigned r = min(g * kRouterRows + i, unsigned{moe::kExperts});
      const uint4* row =
          reinterpret_cast<const uint4*>(p.w + std::size_t{r} * moe::kHidden + lane * 32);
#pragma unroll
      for (int c = 0; c < 4; ++c)
        v[i][c] = row[c];
    }
    const unsigned t_end = min(t0 + kRouterTok, p.n_tok);
    for (unsigned t = t0; t < t_end; ++t) {
      const float* x = p.x + std::size_t{t} * moe::kHidden + lane * 32;
#pragma unroll
      for (int i = 0; i < kRouterRows; ++i) {
        const unsigned r = g * kRouterRows + i;
        const float acc = wave_sum(router_lane_dot(v[i], x));
        if (lane == kWave - 1 && r <= unsigned{moe::kExperts})
          p.logits[std::size_t{t} * (moe::kExperts + 1) + r] = acc;
      }
    }
  }
}

__host__ __device__ inline unsigned moe_router_batch_items(unsigned n_tok, int block) {
  const unsigned waves = static_cast<unsigned>(block / kWave);
  return (moe_detail::kRouterGroups + waves - 1) / waves *
         ((n_tok + moe_detail::kRouterTok - 1) / moe_detail::kRouterTok);
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) moe_router_batch_kernel(MoeRouterBatchParams p) {
  moe_router_batch_op<kBlock>(p, 0, moe_router_batch_items(p.n_tok, kBlock));
}

struct MoeTopkBatchParams {
  const float* logits;  // [n_tok][257]
  int* ids;             // [n_tok][9]
  float* weights;       // [n_tok][9]
  unsigned n_tok;
};

// Tokens [first, first + count), one workgroup each.
template <int kBlock>
__device__ void moe_topk_batch_op(const MoeTopkBatchParams& p, unsigned first, unsigned count) {
  __shared__ int sid[moe::kTopK];
  __shared__ float sw[moe::kTopK];
  for (unsigned t = first + blockIdx.x; t < first + count; t += gridDim.x) {
    const float* l = p.logits + std::size_t{t} * (moe::kExperts + 1);
    moe_topk<kBlock>(l, sid, sw);
    if (threadIdx.x < moe::kTopK) {
      p.ids[t * moe::kSlots + threadIdx.x] = sid[threadIdx.x];
      p.weights[t * moe::kSlots + threadIdx.x] = sw[threadIdx.x];
    } else if (threadIdx.x == moe::kTopK) {
      p.ids[t * moe::kSlots + moe::kTopK] = moe::kExperts;
      p.weights[t * moe::kSlots + moe::kTopK] = 1.0f / (1.0f + expf(-l[moe::kExperts]));
    }
    __syncthreads();
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) moe_topk_batch_kernel(MoeTopkBatchParams p) {
  moe_topk_batch_op<kBlock>(p, 0, p.n_tok);
}

struct MoeGroupParams {
  const int* ids;  // [n_tok][9]
  int* order;      // [9 n_tok]: assignments sorted by expert
  int4* tasks;     // {expert, first index into order, count, 0}
  int* n_tasks;
  unsigned n_tok;
};

__host__ __device__ inline unsigned moe_max_tasks(unsigned n_tok) {
  return n_tok * moe::kSlots / moe::kTaskTok + moe::kExperts + 1;
}

// One workgroup.
template <int kBlock>
__device__ void moe_group_op(const MoeGroupParams& p) {
  constexpr int kE = moe::kExperts + 1;
  __shared__ int cnt[kE], start[kE], tstart[kE + 1], cursor[kE];
  const unsigned n = p.n_tok * moe::kSlots;
  for (int e = threadIdx.x; e < kE; e += kBlock)
    cnt[e] = 0, cursor[e] = 0;
  __syncthreads();
  for (unsigned a = threadIdx.x; a < n; a += kBlock)
    atomicAdd(&cnt[p.ids[a]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0, ts = 0;
    for (int e = 0; e < kE; ++e) {
      start[e] = s, tstart[e] = ts;
      s += cnt[e], ts += (cnt[e] + moe::kTaskTok - 1) / moe::kTaskTok;
    }
    tstart[kE] = ts;
    *p.n_tasks = ts;
  }
  __syncthreads();
  for (unsigned a = threadIdx.x; a < n; a += kBlock) {
    const int e = p.ids[a];
    p.order[start[e] + atomicAdd(&cursor[e], 1)] = static_cast<int>(a);
  }
  for (int e = threadIdx.x; e < kE; e += kBlock) {
    for (int i = 0; tstart[e] + i < tstart[e + 1]; ++i) {
      const int c = min(moe::kTaskTok, cnt[e] - moe::kTaskTok * i);
      p.tasks[tstart[e] + i] = int4{e, start[e] + moe::kTaskTok * i, c, 0};
    }
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) moe_group_kernel(MoeGroupParams p) {
  moe_group_op<kBlock>(p);
}

namespace moe_detail {

using gemv::half2_t;

// 16 subnormal weight pairs against 16 wave-uniform FP16 activation pairs (4 x uint4).
__device__ inline float dot16(const unsigned (&w)[16], const uint4* x) {
  float s = 0;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 v = x[c];
    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, w[4 * c]),
                               __builtin_bit_cast(half2_t, v.x), s, false);
    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, w[4 * c + 1]),
                               __builtin_bit_cast(half2_t, v.y), s, false);
    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, w[4 * c + 2]),
                               __builtin_bit_cast(half2_t, v.z), s, false);
    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, w[4 * c + 3]),
                               __builtin_bit_cast(half2_t, v.w), s, false);
  }
  return s;
}

// One nibble group (64 weights: sub-blocks 2g and 2g+1) of a Q4_K r1 row, as subnormal pairs in
// natural element order (q 2^-24, exact).
struct Q4kChunk {
  unsigned lo[16], hi[16];
  float d, sc0, sc1, m0, m1;  // d scaled by 2^24; m0, m1 multiplied by dmin
};

__device__ inline Q4kChunk q4k_chunk(const std::uint8_t* blk, const uint4& h, int g) {
  Q4kChunk c;
  const uint4 q0 = *reinterpret_cast<const uint4*>(blk + 16 + 32 * g);
  const uint4 q1 = *reinterpret_cast<const uint4*>(blk + 32 + 32 * g);
  const unsigned w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    c.lo[2 * i] = (w[i] & 0xFu) | ((w[i] << 8) & 0xF0000u);
    c.lo[2 * i + 1] = ((w[i] >> 16) & 0xFu) | ((w[i] >> 8) & 0xF0000u);
    c.hi[2 * i] = ((w[i] >> 4) & 0xFu) | ((w[i] << 4) & 0xF0000u);
    c.hi[2 * i + 1] = ((w[i] >> 20) & 0xFu) | ((w[i] >> 12) & 0xF0000u);
  }
  const q4k_detail::Scales s = q4k_detail::r1_scales(h, g);
  const float dmin = q4k_detail::fp16_bits(h.x >> 16);
  c.d = q4k_detail::fp16_bits(h.x & 0xFFFFu) * 0x1p24f;
  c.sc0 = s.sc0, c.sc1 = s.sc1, c.m0 = dmin * s.m0, c.m1 = dmin * s.m1;
  return c;
}

// x: the chunk's 64 activations (8 x uint4); s_lo, s_hi: sums of the two 32-groups.
__device__ inline float q4k_chunk_dot(const Q4kChunk& c, const uint4* x, float s_lo, float s_hi) {
  return c.d * (c.sc0 * dot16(c.lo, x) + c.sc1 * dot16(c.hi, x + 4)) - (c.m0 * s_lo + c.m1 * s_hi);
}

}  // namespace moe_detail

struct MoeGateUpGroupedParams {
  const std::uint8_t* gate_exps;  // Q4_K r1, [256][512 rows], K = 2048
  const std::uint8_t* up_exps;
  const std::uint8_t* gate_sh;  // Q4_K r1, [512 rows]
  const std::uint8_t* up_sh;
  const _Float16* xh;  // [n_tok][2048]
  const float* xs;     // [n_tok][64]: per-32 sums of xh
  const int* order;
  const int4* tasks;
  const int* n_tasks;
  _Float16* ah;  // [9 n_tok][512]: SiLU(gate) * up, rounded to FP16
  float* as;     // [9 n_tok][32]: per-16 sums of ah
};

// Work items [first, first + count) of 2 per task (256 row pairs each), up to 2 * n_tasks.
template <int kBlock>
__device__ void moe_gate_up_grouped_op(const MoeGateUpGroupedParams& p, unsigned first,
                                       unsigned count) {
  using namespace moe_detail;
  static_assert(2 * kBlock == moe::kFf);
  constexpr unsigned kRowBytes = moe::kHidden / 256 * 144;
  const unsigned end = min(first + count, 2u * static_cast<unsigned>(*p.n_tasks));
  for (unsigned item = first + blockIdx.x; item < end; item += gridDim.x) {
    const int4 tk = p.tasks[item >> 1];
    const unsigned j = (item & 1) * kBlock + threadIdx.x;
    const std::uint8_t* gw = tk.x < moe::kExperts
                                 ? p.gate_exps + (std::size_t(tk.x) * moe::kFf + j) * kRowBytes
                                 : p.gate_sh + std::size_t{j} * kRowBytes;
    const std::uint8_t* uw = tk.x < moe::kExperts
                                 ? p.up_exps + (std::size_t(tk.x) * moe::kFf + j) * kRowBytes
                                 : p.up_sh + std::size_t{j} * kRowBytes;
    int tok[moe::kTaskTok];
#pragma unroll
    for (int i = 0; i < moe::kTaskTok; ++i)
      tok[i] = i < tk.z ? p.order[tk.y + i] / moe::kSlots : 0;
    float ag[moe::kTaskTok] = {}, au[moe::kTaskTok] = {};
    for (int b = 0; b < moe::kHidden / 256; ++b) {
      const uint4 hg = *reinterpret_cast<const uint4*>(gw + 144 * b);
      const uint4 hu = *reinterpret_cast<const uint4*>(uw + 144 * b);
#pragma unroll 1
      for (int g = 0; g < 4; ++g) {
        const Q4kChunk cg = q4k_chunk(gw + 144 * b, hg, g), cu = q4k_chunk(uw + 144 * b, hu, g);
        const unsigned k0 = 256 * b + 64 * g;
#pragma unroll
        for (int i = 0; i < moe::kTaskTok; ++i) {
          if (i < tk.z) {
            const uint4* x =
                reinterpret_cast<const uint4*>(p.xh + std::size_t(tok[i]) * moe::kHidden + k0);
            const float2 s = *reinterpret_cast<const float2*>(
                p.xs + std::size_t(tok[i]) * (moe::kHidden / 32) + k0 / 32);
            ag[i] += q4k_chunk_dot(cg, x, s.x, s.y);
            au[i] += q4k_chunk_dot(cu, x, s.x, s.y);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < moe::kTaskTok; ++i) {
      if (i < tk.z) {
        const int a = p.order[tk.y + i];
        const _Float16 h = static_cast<_Float16>(ag[i] / (1.0f + expf(-ag[i])) * au[i]);
        p.ah[std::size_t(a) * moe::kFf + j] = h;
        const float s = moe_detail::row16_sum(static_cast<float>(h));
        if ((j & 15) == 0)
          p.as[std::size_t(a) * (moe::kFf / 16) + j / 16] = s;
      }
    }
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) moe_gate_up_grouped_kernel(MoeGateUpGroupedParams p) {
  moe_gate_up_grouped_op<kBlock>(p, 0, ~0u);
}

struct MoeDownGroupedParams {
  const std::uint8_t* down_exps;  // r1, [256][2048 rows], K = 512
  const std::uint8_t* down_sh;    // r1, [2048 rows]
  const _Float16* ah;             // [9 n_tok][512]
  const float* as;                // [9 n_tok][32]
  const float* weights;           // [9 n_tok]
  const int* order;
  const int4* tasks;
  const int* n_tasks;
  float* part;  // [9 n_tok][2048]: weight * down(ah)
};

// Work items [first, first + count) of 8 per task (256 output rows each), up to 8 * n_tasks.
template <QType T, int kBlock>
__device__ void moe_down_grouped_op(const MoeDownGroupedParams& p, unsigned first, unsigned count) {
  using namespace moe_detail;
  constexpr unsigned kParts = moe::kHidden / kBlock;
  constexpr unsigned nb = moe::kFf / 256;
  constexpr unsigned kRowBytes = T == QType::Q4K ? nb * 144 : (nb * 210 + 15) / 16 * 16;
  const unsigned end = min(first + count, kParts * static_cast<unsigned>(*p.n_tasks));
  for (unsigned item = first + blockIdx.x; item < end; item += gridDim.x) {
    const int4 tk = p.tasks[item / kParts];
    const unsigned r = item % kParts * kBlock + threadIdx.x;
    const std::uint8_t* row = tk.x < moe::kExperts
                                  ? p.down_exps + (std::size_t(tk.x) * moe::kHidden + r) * kRowBytes
                                  : p.down_sh + std::size_t{r} * kRowBytes;
    int as_[moe::kTaskTok];
#pragma unroll
    for (int i = 0; i < moe::kTaskTok; ++i)
      as_[i] = i < tk.z ? p.order[tk.y + i] : 0;
    float acc[moe::kTaskTok] = {};
    for (unsigned b = 0; b < nb; ++b) {
      if constexpr (T == QType::Q4K) {
        const uint4 h = *reinterpret_cast<const uint4*>(row + 144 * b);
#pragma unroll 1
        for (int g = 0; g < 4; ++g) {
          const Q4kChunk c = q4k_chunk(row + 144 * b, h, g);
          const unsigned k0 = 256 * b + 64 * g;
#pragma unroll
          for (int i = 0; i < moe::kTaskTok; ++i) {
            if (i < tk.z) {
              const uint4* x =
                  reinterpret_cast<const uint4*>(p.ah + std::size_t(as_[i]) * moe::kFf + k0);
              const float4 s = *reinterpret_cast<const float4*>(
                  p.as + std::size_t(as_[i]) * (moe::kFf / 16) + k0 / 16);
              acc[i] += q4k_chunk_dot(c, x, s.x + s.y, s.z + s.w);
            }
          }
        }
      } else {
        const float d =
            q4k_detail::fp16_bits(*reinterpret_cast<const unsigned short*>(row + nb * 208 + 2 * b));
#pragma unroll 1
        for (unsigned ss = 0; ss < 4; ++ss) {
          // Elements 128 hh + 64 u + l: ql byte l of half hh (nibble u), qh byte l % 32 (bits
          // 4 u + 2 (l / 32)), scale 8 hh + 4 u + l / 16.
          const unsigned hh = ss >> 1, u = ss & 1;
          const uint4* lp = reinterpret_cast<const uint4*>(row + 128 * b + 64 * hh);
          const uint4* hp = reinterpret_cast<const uint4*>(row + nb * 128 + 64 * b + 32 * hh);
          const uint4 l4[4] = {lp[0], lp[1], lp[2], lp[3]}, h4[2] = {hp[0], hp[1]};
          const unsigned scw =
              *reinterpret_cast<const unsigned*>(row + nb * 192 + 16 * b + 8 * hh + 4 * u);
          const unsigned lw[16] = {l4[0].x, l4[0].y, l4[0].z, l4[0].w, l4[1].x, l4[1].y,
                                   l4[1].z, l4[1].w, l4[2].x, l4[2].y, l4[2].z, l4[2].w,
                                   l4[3].x, l4[3].y, l4[3].z, l4[3].w};
          const unsigned hw[8] = {h4[0].x, h4[0].y, h4[0].z, h4[0].w,
                                  h4[1].x, h4[1].y, h4[1].z, h4[1].w};
          unsigned w[4][8];  // [16-group][pair]
#pragma unroll
          for (int c = 0; c < 16; ++c) {
            const unsigned v = ((lw[c] >> (4 * u)) & 0x0F0F0F0Fu) |
                               (((hw[c & 7] >> (4 * u + 2 * (c >> 3))) & 0x03030303u) << 4);
            w[c >> 2][(2 * c) & 7] = (v & 0xFFu) | ((v << 8) & 0xFF0000u);
            w[c >> 2][(2 * c + 1) & 7] = ((v >> 16) & 0xFFu) | ((v >> 8) & 0xFF0000u);
          }
          float scale[4];
#pragma unroll
          for (int q = 0; q < 4; ++q)
            scale[q] = d * 0x1p24f * static_cast<float>(static_cast<std::int8_t>(scw >> (8 * q)));
          const unsigned k0 = 256 * b + 64 * ss;
#pragma unroll
          for (int i = 0; i < moe::kTaskTok; ++i) {
            if (i < tk.z) {
              const uint4* x =
                  reinterpret_cast<const uint4*>(p.ah + std::size_t(as_[i]) * moe::kFf + k0);
              const float4 s = *reinterpret_cast<const float4*>(
                  p.as + std::size_t(as_[i]) * (moe::kFf / 16) + k0 / 16);
              const float s16[4] = {s.x, s.y, s.z, s.w};
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                float v = -32.0f * 0x1p-24f * s16[q];
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                  const uint4 xv = x[2 * q + c];
                  v = __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, w[q][4 * c]),
                                             __builtin_bit_cast(half2_t, xv.x), v, false);
                  v = __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, w[q][4 * c + 1]),
                                             __builtin_bit_cast(half2_t, xv.y), v, false);
                  v = __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, w[q][4 * c + 2]),
                                             __builtin_bit_cast(half2_t, xv.z), v, false);
                  v = __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, w[q][4 * c + 3]),
                                             __builtin_bit_cast(half2_t, xv.w), v, false);
                }
                acc[i] += scale[q] * v;
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < moe::kTaskTok; ++i) {
      if (i < tk.z)
        p.part[std::size_t(as_[i]) * moe::kHidden + r] = p.weights[as_[i]] * acc[i];
    }
  }
}

template <QType T, int kBlock>
__global__ void __launch_bounds__(kBlock) moe_down_grouped_kernel(MoeDownGroupedParams p) {
  moe_down_grouped_op<T, kBlock>(p, 0, ~0u);
}

struct MoeCombineParams {
  const float* part;  // [9 n_tok][2048], weighted slot outputs
  float* out;         // [n_tok][2048]
  unsigned n_tok;
};

// Float4 elements [first, first + count) of n_tok * 512.
__device__ inline void moe_combine_op(const MoeCombineParams& p, unsigned first, unsigned count) {
  constexpr unsigned kRow = moe::kHidden / 4;
  for (unsigned i = first + blockIdx.x * blockDim.x + threadIdx.x; i < first + count;
       i += gridDim.x * blockDim.x) {
    const unsigned t = i / kRow, c = i % kRow;
    const float4* src =
        reinterpret_cast<const float4*>(p.part) + std::size_t{t} * moe::kSlots * kRow + c;
    float4 s = src[0];
#pragma unroll
    for (int k = 1; k < moe::kSlots; ++k) {
      const float4 v = src[std::size_t(k) * kRow];
      s.x += v.x, s.y += v.y, s.z += v.z, s.w += v.w;
    }
    reinterpret_cast<float4*>(p.out)[i] = s;
  }
}

__global__ void __launch_bounds__(256) moe_combine_kernel(MoeCombineParams p) {
  moe_combine_op(p, 0, p.n_tok * (moe::kHidden / 4));
}

}  // namespace miso::kernels
