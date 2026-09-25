#pragma once

// Sparse MoE block (qwen3_5_moe), decode path, in three launches:
//   moe_router:  logits = W_r x for 256 experts plus the shared-expert gate (row 256), BF16
//   weights. moe_gate_up: every workgroup re-derives the top-8 experts from the logits (no extra
//   launch),
//                then computes gate and up (512 each) for the 8 routed experts and the shared one.
//   moe_down:    out = sum_k w_k down_k(SiLU(gate_k) * up_k) + sigmoid(g_sh) down_sh(...), with 16
//                lanes per K = 512 row, so one wavefront covers 4 experts of an output row at once.
// Routing weights are the top-8 softmax probabilities renormalised over the 8.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "q4k_gemv.hpp"
#include "q6k_gemv.hpp"
#include "wave.hpp"

namespace miso::kernels {

namespace moe {
constexpr int kExperts = 256, kTopK = 8, kSlots = kTopK + 1, kHidden = 2048, kFf = 512;
}  // namespace moe

struct MoeRouterParams {
  const std::uint16_t* w;  // [257][2048] BF16: 256 router rows, then the shared-expert gate
  const float* x;          // [2048]
  float* logits;           // [257]
};

// Rows [first, first + count) of 257; one wavefront per row, 32 contiguous weights per lane.
template <int kBlock>
__device__ void moe_router_op(const MoeRouterParams& p, unsigned first, unsigned count) {
  const int lane = threadIdx.x % kWave;
  const unsigned wave = blockIdx.x * (kBlock / kWave) + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * (kBlock / kWave);
  const float* x = p.x + lane * 32;
  for (unsigned r = first + wave; r < first + count; r += n_waves) {
    const uint4* row =
        reinterpret_cast<const uint4*>(p.w + std::size_t{r} * moe::kHidden + lane * 32);
    float acc = 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 v = row[c];
      const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc += __builtin_bit_cast(float, w[e] << 16) * x[8 * c + 2 * e];
        acc += __builtin_bit_cast(float, w[e] & 0xFFFF0000u) * x[8 * c + 2 * e + 1];
      }
    }
    acc = wave_sum(acc);
    if (lane == kWave - 1)
      p.logits[r] = acc;
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) moe_router_kernel(MoeRouterParams p) {
  moe_router_op<kBlock>(p, 0, moe::kExperts + 1);
}

// Top-8 of the 256 router logits, by rank (ties broken by lower index), with renormalised softmax
// weights. Needs all 256 threads of the workgroup.
template <int kBlock>
__device__ inline void moe_topk(const float* logits, int* ids, float* weights) {
  static_assert(kBlock == moe::kExperts);
  __shared__ float l[moe::kExperts], red[kBlock / kWave];
  const unsigned t = threadIdx.x;
  const float mine = logits[t];
  l[t] = mine;
  __syncthreads();
  unsigned rank = 0;
  for (unsigned u = 0; u < moe::kExperts; ++u)
    rank += l[u] > mine || (l[u] == mine && u < t);
  const float mx = block_max<kBlock>(mine, red);
  const float e = rank < moe::kTopK ? expf(mine - mx) : 0.0f;
  const float sum = block_sum<kBlock>(e, red);
  if (rank < moe::kTopK)
    ids[rank] = static_cast<int>(t), weights[rank] = e / sum;
  __syncthreads();
}

struct MoeGateUpParams {
  const float* logits;            // [257]
  const std::uint8_t* gate_exps;  // Q4_K r1, [256 experts][512 rows], K = 2048
  const std::uint8_t* up_exps;
  const std::uint8_t* gate_sh;  // Q4_K r1, [512 rows]
  const std::uint8_t* up_sh;
  const float* x;  // [2048]
  float* h;        // [9 slots][1024]: 512 gate then 512 up; slot 8 = shared
  int* ids;        // [8], written by workgroup 0
  float* weights;  // [8]
};

// Rows [first, first + count) of 9 * 1024.
template <int kBlock>
__device__ void moe_gate_up_op(const MoeGateUpParams& p, unsigned first, unsigned count) {
  using namespace q4k_detail;
  constexpr unsigned kRowBytes = moe::kHidden / 256 * 144;
  __shared__ int sid[moe::kTopK];
  __shared__ float sw[moe::kTopK];
  moe_topk<kBlock>(p.logits, sid, sw);
  if (blockIdx.x == 0 && threadIdx.x < moe::kTopK) {
    p.ids[threadIdx.x] = sid[threadIdx.x], p.weights[threadIdx.x] = sw[threadIdx.x];
  }
  const int lane = threadIdx.x % kWave;
  const unsigned wave = blockIdx.x * (kBlock / kWave) + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * (kBlock / kWave);
  const SlotAct<ActFormat::Fp16> act = load_slot<ActFormat::Fp16>(p.x, lane);
  for (unsigned r = first + wave; r < first + count; r += n_waves) {
    const unsigned slot = r / (2 * moe::kFf), j = r % (2 * moe::kFf);
    const bool up = j >= moe::kFf;
    const std::size_t row = j % moe::kFf;
    const std::uint8_t* base =
        slot < moe::kTopK
            ? (up ? p.up_exps : p.gate_exps) + (std::size_t(sid[slot]) * moe::kFf + row) * kRowBytes
            : (up ? p.up_sh : p.gate_sh) + row * kRowBytes;
    float acc = slot_dot<ActFormat::Fp16>(load_w(base, lane), lane, act);
    acc = wave_sum(acc);
    if (lane == kWave - 1)
      p.h[r] = acc;
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) moe_gate_up_kernel(MoeGateUpParams p) {
  moe_gate_up_op<kBlock>(p, 0, moe::kSlots * 2 * moe::kFf);
}

enum class QType { Q4K, Q6K };

struct MoeDownParams {
  const float* logits;            // [257]; logits[256] gates the shared expert
  const int* ids;                 // [8]
  const float* weights;           // [8]
  const std::uint8_t* down_exps;  // r1, [256 experts][2048 rows], K = 512
  const std::uint8_t* down_sh;    // r1, [2048 rows]
  const float* h;                 // [9][1024] from moe_gate_up
  float* out;                     // [2048]
};

namespace moe_detail {

// Sum within each 16-lane DPP row; every lane of the row gets the row's sum.
__device__ inline float row16_sum(float v) {
  v += dpp<0xB1>(v);
  v += dpp<0x4E>(v);
  v += dpp<0x141>(v);
  v += dpp<0x140>(v);
  return v;
}

template <QType T>
struct Down;

template <>
struct Down<QType::Q4K> {
  static constexpr unsigned kRowBytes = moe::kFf / 256 * 144;
  using Act = q4k_detail::SlotAct<ActFormat::Fp16>;
  static __device__ Act act(const float* a, int slot) {
    return q4k_detail::load_slot<ActFormat::Fp16>(a, slot);
  }
  static __device__ float dot(const std::uint8_t* row, int slot, const Act& a) {
    return q4k_detail::slot_dot<ActFormat::Fp16>(q4k_detail::load_w(row, slot), slot, a);
  }
};

template <>
struct Down<QType::Q6K> {
  static constexpr unsigned kRowBytes = (moe::kFf / 256 * 210 + 15) / 16 * 16;
  using Act = q6k_detail::SlotAct;
  static __device__ Act act(const float* a, int slot) { return q6k_detail::load_slot(a, slot); }
  static __device__ float dot(const std::uint8_t* row, int slot, const Act& a) {
    return q6k_detail::slot_dot(q6k_detail::load_w<moe::kFf>(row, slot), slot, a);
  }
};

}  // namespace moe_detail

// Output rows [first, first + count) of 2048. Lane group g = lane / 16 handles slot 4 it + g in
// step it (it = 0..2 covers the 9 slots).
template <QType T, int kBlock>
__device__ void moe_down_op(const MoeDownParams& p, unsigned first, unsigned count) {
  using D = moe_detail::Down<T>;
  constexpr int kSteps = (moe::kSlots + 3) / 4;
  __shared__ float a[moe::kSlots][moe::kFf];
  __shared__ int sid[moe::kTopK];
  __shared__ float coef[moe::kSlots];
  for (unsigned i = threadIdx.x; i < moe::kSlots * moe::kFf; i += kBlock) {
    const unsigned s = i / moe::kFf, k = i % moe::kFf;
    const float g = p.h[s * 2 * moe::kFf + k], u = p.h[s * 2 * moe::kFf + moe::kFf + k];
    a[s][k] = g / (1.0f + expf(-g)) * u;
  }
  if (threadIdx.x < moe::kTopK)
    sid[threadIdx.x] = p.ids[threadIdx.x], coef[threadIdx.x] = p.weights[threadIdx.x];
  if (threadIdx.x == 0)
    coef[moe::kTopK] = 1.0f / (1.0f + expf(-p.logits[moe::kExperts]));
  __syncthreads();

  const int lane = threadIdx.x % kWave, grp = lane / 16, l16 = lane % 16;
  const unsigned wave = blockIdx.x * (kBlock / kWave) + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * (kBlock / kWave);
  typename D::Act act[kSteps];
#pragma unroll
  for (int it = 0; it < kSteps; ++it) {
    const int s = min(4 * it + grp, moe::kSlots - 1);
    act[it] = D::act(a[s], l16);
  }
  for (unsigned r = first + wave; r < first + count; r += n_waves) {
    float v = 0;
#pragma unroll
    for (int it = 0; it < kSteps; ++it) {
      const int s = 4 * it + grp, sc = min(s, moe::kSlots - 1);
      const std::uint8_t* base =
          sc < moe::kTopK ? p.down_exps + (std::size_t(sid[sc]) * moe::kHidden + r) * D::kRowBytes
                          : p.down_sh + std::size_t{r} * D::kRowBytes;
      const float part = moe_detail::row16_sum(D::dot(base, l16, act[it]));
      v += s < moe::kSlots ? coef[sc] * part : 0.0f;
    }
    v += dpp<0x142, 0xA>(v);  // rows 1, 3 += rows 0, 2
    v += dpp<0x143, 0xC>(v);  // rows 2, 3 += row 1: lane 63 holds the total
    if (lane == kWave - 1)
      p.out[r] = v;
  }
}

template <QType T, int kBlock>
__global__ void __launch_bounds__(kBlock) moe_down_kernel(MoeDownParams p) {
  moe_down_op<T, kBlock>(p, 0, moe::kHidden);
}

}  // namespace miso::kernels
