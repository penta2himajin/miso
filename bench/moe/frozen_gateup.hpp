#pragma once

// Frozen top-k and gate/up operators from main 438fb45 (kernels/moe_decode.hpp).
// Benchmark-only control: retains redundant per-workgroup coefficient computation.
#include "moe_decode.hpp"

namespace miso::coeff_bench::frozen {
using namespace miso::kernels;

// Top-8 of the 256 router logits (ties → lower index), renormalised softmax weights.
// Wave 0 owns selection: each of its 64 lanes holds four logits (lane + 64*j), and eight
// wave-local argmaxes replace the previous eight block-wide reductions (peer review loop3).
template <int kBlock>
__device__ inline void moe_topk(const float* logits, int* ids, float* weights) {
  static_assert(kBlock == moe::kExperts);
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  if (wave == 0) {
    float v[4];
    int idx[4];
    bool alive[4];
    int pi[moe::kTopK];
    float pv[moe::kTopK];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      idx[j] = lane + kWave * j;
      v[j] = logits[idx[j]];
      alive[j] = true;
    }
    for (int k = 0; k < moe::kTopK; ++k) {
      float local_mx = -__builtin_inff();
      int local_i = moe::kExperts;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (alive[j] && (v[j] > local_mx || (v[j] == local_mx && idx[j] < local_i))) {
          local_mx = v[j];
          local_i = idx[j];
        }
      }
      const float mx = __shfl(wave_max(local_mx), kWave - 1);
      const float idx_f =
          (local_mx == mx) ? static_cast<float>(local_i) : static_cast<float>(moe::kExperts);
      const float best_f = __shfl(wave_reduce(
                                      idx_f, [](float a, float b) { return fminf(a, b); },
                                      static_cast<float>(moe::kExperts)),
                                  kWave - 1);
      const int best = static_cast<int>(best_f);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (idx[j] == best)
          alive[j] = false;
      pi[k] = best;
      pv[k] = logits[best];
    }
    float mx = pv[0];
#pragma unroll
    for (int k = 1; k < moe::kTopK; ++k)
      mx = fmaxf(mx, pv[k]);
    float e = 0;
#pragma unroll
    for (int k = 0; k < moe::kTopK; ++k)
      e += lane == k ? expf(pv[k] - mx) : 0.0f;
    float sum = e;
    sum += __shfl_xor(sum, 4);
    sum += __shfl_xor(sum, 2);
    sum += __shfl_xor(sum, 1);
    if (lane < moe::kTopK)
      ids[lane] = pi[lane], weights[lane] = expf(pv[lane] - mx) / sum;
  }
  __syncthreads();
}

// Rows [first, first + count) of 9 * 512. One wavefront writes SiLU(gate_row) * up_row.
template <int kBlock>
__device__ void moe_gate_up_op(const MoeGateUpParams& p, unsigned first, unsigned count) {
  using namespace q4k_detail;
  constexpr unsigned kRowBytes = moe::kHidden / 256 * 144;
  __shared__ int sid[moe::kTopK];
  __shared__ float sw[moe::kTopK];
  moe_topk<kBlock>(p.logits, sid, sw);
  if (blockIdx.x == 0 && threadIdx.x < moe::kTopK)
    p.ids[threadIdx.x] = sid[threadIdx.x], p.weights[threadIdx.x] = sw[threadIdx.x];
  const int lane = threadIdx.x % kWave;
  const unsigned wave = blockIdx.x * (kBlock / kWave) + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * (kBlock / kWave);
  const SlotAct<ActFormat::Fp16> act = load_slot<ActFormat::Fp16>(p.x, lane);
  auto rows = [&](unsigned r) {
    const unsigned slot = r / moe::kFf, row = r % moe::kFf;
    const std::uint8_t* gate =
        slot < moe::kTopK ? p.gate_exps + (std::size_t(sid[slot]) * moe::kFf + row) * kRowBytes
                          : p.gate_sh + row * kRowBytes;
    const std::uint8_t* up = slot < moe::kTopK
                                 ? p.up_exps + (std::size_t(sid[slot]) * moe::kFf + row) * kRowBytes
                                 : p.up_sh + row * kRowBytes;
    return std::pair<const std::uint8_t*, const std::uint8_t*>{gate, up};
  };
  auto emit = [&](unsigned r, const SlotW& gw, const SlotW& uw) {
    const float g = wave_sum(slot_dot<ActFormat::Fp16>(gw, lane, act));
    const float u = wave_sum(slot_dot<ActFormat::Fp16>(uw, lane, act));
    if (lane == kWave - 1)
      p.h[r] = g / (1.0f + expf(-g)) * u;
  };
  unsigned r = first + wave;
  if (r < first + count) {
    SlotW gw = load_w(rows(r).first, lane), uw = load_w(rows(r).second, lane);
    for (unsigned nxt = r + n_waves; nxt < first + count; nxt += n_waves) {
      const auto p2 = rows(nxt);
      const SlotW g2 = load_w(p2.first, lane), u2 = load_w(p2.second, lane);
      emit(r, gw, uw);
      gw = g2, uw = u2, r = nxt;
    }
    emit(r, gw, uw);
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) gate_up_kernel(MoeGateUpParams p) {
  miso::coeff_bench::frozen::moe_gate_up_op<kBlock>(p, 0, moe::kSlots * moe::kFf);
}

}  // namespace miso::coeff_bench::frozen
