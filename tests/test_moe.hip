#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/moe.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "moe_decode.hpp"

using miso::gguf::File;

namespace {

constexpr unsigned kHidden = 2048;
namespace kernels = miso::kernels;

// Frozen main438fb45 top-k: every workgroup computes coefficients. Test-only reference keeps
// selection, coefficient reduction and exp operation order independent of the candidate.
template <int kBlock>
__device__ void frozen_topk(const float* logits, int* ids, float* weights) {
  using namespace kernels;
  static_assert(kBlock == moe::kExperts);
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave;
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
          local_mx == mx ? static_cast<float>(local_i) : static_cast<float>(moe::kExperts);
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

constexpr unsigned kTopkGrid = 120, kTopkStride = 10;

template <bool kCandidate, bool kDefaultWeights = false>
__global__ void topk_grid(const float* logits, int* ids, float* weights) {
  const unsigned at = blockIdx.x * kTopkStride + 1;
  if constexpr (!kCandidate)
    frozen_topk<256>(logits, ids + at, weights + at);
  else if constexpr (kDefaultWeights)
    kernels::moe_topk<256>(logits, ids + at, weights + at);
  else
    kernels::moe_topk<256>(logits, ids + at, weights + at, blockIdx.x == 0);
}

void check_topk_grid(const std::vector<float>& logits) {
  constexpr int kIdCanary = -1234567;
  constexpr float kWeightCanary = -9123.5f;
  const std::vector<int> id_initial(kTopkGrid * kTopkStride, kIdCanary);
  const std::vector<float> weight_initial(kTopkGrid * kTopkStride, kWeightCanary);
  miso::DeviceBuffer<float> dl(logits), ref_weights(weight_initial);
  miso::DeviceBuffer<int> ref_ids(id_initial);
  topk_grid<false><<<kTopkGrid, 256>>>(dl.data(), ref_ids.data(), ref_weights.data());
  MISO_HIP_CHECK(hipGetLastError());
  const auto want_ids = ref_ids.to_host();
  const auto want_weights = ref_weights.to_host();
  for (const bool default_weights : {false, true}) {
    CAPTURE(default_weights);
    miso::DeviceBuffer<float> weights(weight_initial);
    miso::DeviceBuffer<int> ids(id_initial);
    if (default_weights)
      topk_grid<true, true><<<kTopkGrid, 256>>>(dl.data(), ids.data(), weights.data());
    else
      topk_grid<true><<<kTopkGrid, 256>>>(dl.data(), ids.data(), weights.data());
    MISO_HIP_CHECK(hipGetLastError());
    const auto got_ids = ids.to_host();
    const auto got_weights = weights.to_host();
    for (unsigned block = 0; block < kTopkGrid; ++block) {
      CAPTURE(block);
      for (unsigned slot = 0; slot < kTopkStride; ++slot) {
        const unsigned at = block * kTopkStride + slot;
        CHECK(got_ids[at] == want_ids[at]);
        const bool guard = slot == 0 || slot == kTopkStride - 1;
        const float want =
            guard || (!default_weights && block != 0) ? kWeightCanary : want_weights[at];
        CHECK(std::bit_cast<std::uint32_t>(got_weights[at]) == std::bit_cast<std::uint32_t>(want));
      }
    }
  }
}

// Frozen old gate/up wrapper: full coefficients in every workgroup and original row scheduling.
__global__ void __launch_bounds__(256) frozen_gate_up(kernels::MoeGateUpParams p) {
  using namespace kernels;
  using namespace q4k_detail;
  constexpr unsigned kRowBytes = moe::kHidden / 256 * 144;
  __shared__ int sid[moe::kTopK];
  __shared__ float sw[moe::kTopK];
  frozen_topk<256>(p.logits, sid, sw);
  if (blockIdx.x == 0 && threadIdx.x < moe::kTopK)
    p.ids[threadIdx.x] = sid[threadIdx.x], p.weights[threadIdx.x] = sw[threadIdx.x];
  const int lane = threadIdx.x % kWave;
  const unsigned wave = blockIdx.x * 4 + threadIdx.x / kWave;
  const unsigned n_waves = gridDim.x * 4;
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
  unsigned r = wave;
  if (r < moe::kSlots * moe::kFf) {
    SlotW gw = load_w(rows(r).first, lane), uw = load_w(rows(r).second, lane);
    for (unsigned nxt = r + n_waves; nxt < moe::kSlots * moe::kFf; nxt += n_waves) {
      const auto p2 = rows(nxt);
      const SlotW g2 = load_w(p2.first, lane), u2 = load_w(p2.second, lane);
      emit(r, gw, uw);
      gw = g2, uw = u2, r = nxt;
    }
    emit(r, gw, uw);
  }
}

std::vector<float> golden_f32(const File& g, const std::string& name) {
  const auto* t = g.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  const auto b = g.tensor_data(*t);
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

}  // namespace

TEST_CASE("MoE coefficient-once preserves every workgroup's IDs and skips other weight writes") {
  std::vector<float> logits(257, 0.0f);
  check_topk_grid(logits);  // all ties select lower indices
  for (unsigned i = 0; i < 256; ++i)
    logits[i] = static_cast<int>(i % 11) - 5;  // ties across all four lane-owned quadrants
  logits[256] = 100000.0f;                     // shared-expert gate must not participate in top-k
  check_topk_grid(logits);
  for (unsigned i = 0; i < 256; ++i)
    logits[i] = i % 2 == 0 ? -10000.0f - i : 10000.0f + i * 100.0f;
  check_topk_grid(logits);  // finite extremes and underflowing coefficient exponentials
  for (unsigned i = 0; i < 256; ++i)
    logits[i] = static_cast<int>((i * 37) % 257) * 0.013f - 2.0f;
  logits[3] = -0.0f;
  logits[129] = 0.0f;
  check_topk_grid(logits);
}

TEST_CASE("MoE coefficient-once gate/up is bit-identical on real layers 0 and 5" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File model = File::open(MISO_ORNITH_GGUF);
  const File golden = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto inputs = golden_f32(golden, "layer0.moe_in");
  miso::DeviceBuffer<float> x(inputs), logits(257);
  constexpr unsigned kOutput = kernels::moe::kSlots * kernels::moe::kFf;
  constexpr float kCanary = -9123.5f;
  for (const int layer : {0, 5}) {
    CAPTURE(layer);
    const auto moe = miso::MoeLayer::load(model, layer);
    for (unsigned token = 0; token < inputs.size() / kHidden; ++token) {
      CAPTURE(token);
      const float* activation = x.data() + token * kHidden;
      const kernels::MoeRouterParams rp{moe.router.data(), activation, logits.data()};
      kernels::moe_router_kernel<256><<<65, 256>>>(rp);
      MISO_HIP_CHECK(hipGetLastError());
      check_topk_grid(logits.to_host());  // exact routing and default full coefficients
      miso::DeviceBuffer<float> got_h(std::vector<float>(kOutput + 2, kCanary));
      miso::DeviceBuffer<float> want_h(std::vector<float>(kOutput + 2, kCanary));
      miso::DeviceBuffer<int> got_ids(8), want_ids(8);
      miso::DeviceBuffer<float> got_w(8), want_w(8);
      kernels::MoeGateUpParams p{logits.data(),
                                 moe.gate_exps.data.data(),
                                 moe.up_exps.data.data(),
                                 moe.gate_sh.data.data(),
                                 moe.up_sh.data.data(),
                                 activation,
                                 got_h.data() + 1,
                                 got_ids.data(),
                                 got_w.data()};
      kernels::moe_gate_up_kernel<256><<<120, 256>>>(p);
      p.h = want_h.data() + 1;
      p.ids = want_ids.data();
      p.weights = want_w.data();
      frozen_gate_up<<<120, 256>>>(p);
      MISO_HIP_CHECK(hipGetLastError());
      const auto actual = got_h.to_host(), expected = want_h.to_host();
      for (unsigned i = 0; i < actual.size(); ++i)
        CHECK(std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected[i]));
      const auto actual_ids = got_ids.to_host(), expected_ids = want_ids.to_host();
      const auto actual_w = got_w.to_host(), expected_w = want_w.to_host();
      for (unsigned i = 0; i < 8; ++i) {
        CHECK(actual_ids[i] == expected_ids[i]);
        CHECK(std::bit_cast<std::uint32_t>(actual_w[i]) ==
              std::bit_cast<std::uint32_t>(expected_w[i]));
      }
    }
  }
}

// For every token of the golden prompt, the MoE block must pick exactly the reference's 8 experts
// in the same order, with the same routing weights, and reproduce its output. Output tolerance
// 1e-3 relative L2: an FP64 run of the reference with FP16-rounded expert inputs (router and
// shared-expert gate kept FP32, as in the engine) deviates by 3.58e-4 / 3.87e-4 on layers 0 / 3,
// which the engine matches.
TEST_CASE("MoE decode reproduces golden routing and output for layers 0 and 3" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  for (const int layer : {0, 3}) {
    CAPTURE(layer);
    const std::string p = "layer" + std::to_string(layer) + ".";
    const auto x = golden_f32(g, p + "moe_in"), want = golden_f32(g, p + "moe_out");
    const auto want_w = golden_f32(g, p + "router.1"), want_id = golden_f32(g, p + "router.2");
    const unsigned n_tok = static_cast<unsigned>(x.size() / kHidden);

    const auto moe = miso::MoeLayer::load(m, layer);
    miso::MoeScratch sc;
    miso::DeviceBuffer<float> dx(kHidden), dy(kHidden);
    double worst_out = 0, worst_w = 0;
    for (unsigned t = 0; t < n_tok; ++t) {
      CAPTURE(t);
      MISO_HIP_CHECK(
          hipMemcpy(dx.data(), x.data() + t * kHidden, kHidden * 4, hipMemcpyHostToDevice));
      miso::moe_decode(moe, sc, dx.data(), dy.data(), nullptr);
      MISO_HIP_CHECK(hipDeviceSynchronize());
      const auto ids = sc.ids.to_host();
      const auto w = sc.weights.to_host();
      for (unsigned k = 0; k < 8; ++k) {
        CAPTURE(k);
        CHECK(ids[k] == static_cast<int>(want_id[t * 8 + k]));
        worst_w = std::max(worst_w, std::fabs(double{w[k]} - want_w[t * 8 + k]));
      }
      const auto y = dy.to_host();
      double num = 0, den = 0;
      for (unsigned i = 0; i < kHidden; ++i) {
        const double d = double{y[i]} - want[t * kHidden + i];
        num += d * d, den += double{want[t * kHidden + i]} * want[t * kHidden + i];
      }
      const double rel = std::sqrt(num / den);
      worst_out = std::max(worst_out, rel);
      CHECK(rel <= 1e-3);
    }
    CHECK(worst_w <= 1e-6);
    MESSAGE("layer " << layer << ": max routing weight error " << worst_w
                     << ", max relative L2 error " << worst_out);
  }
}

// The prefill form routes the whole chunk at once and runs grouped expert GEMMs. Routing must be
// bit-identical to decode (same router dot products, same top-k), and every token must meet the
// decode golden bound. Repeating the prompt 24 times puts up to ~400 assignments on one expert,
// so experts span several tasks; identical inputs must give bit-identical outputs wherever they
// land.
TEST_CASE("MoE prefill routes like decode and reproduces golden output for layers 0 and 3" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  for (const int layer : {0, 3}) {
    CAPTURE(layer);
    const std::string p = "layer" + std::to_string(layer) + ".";
    const auto x = golden_f32(g, p + "moe_in"), want = golden_f32(g, p + "moe_out");
    const auto want_id = golden_f32(g, p + "router.2");
    const unsigned n_prompt = static_cast<unsigned>(x.size() / kHidden);
    const auto moe = miso::MoeLayer::load(m, layer);
    MESSAGE("layer " << layer << " down type "
                     << (moe.down_exps.type == miso::gguf::GgmlType::Q4_K ? "Q4_K" : "Q6_K"));

    // Decode reference: routing ids and weights per token.
    miso::MoeScratch sc;
    miso::DeviceBuffer<float> dx(kHidden), dy(kHidden);
    std::vector<int> dec_ids;
    std::vector<float> dec_w;
    for (unsigned t = 0; t < n_prompt; ++t) {
      MISO_HIP_CHECK(
          hipMemcpy(dx.data(), x.data() + t * kHidden, kHidden * 4, hipMemcpyHostToDevice));
      miso::moe_decode(moe, sc, dx.data(), dy.data(), nullptr);
      const auto ids = sc.ids.to_host();
      const auto w = sc.weights.to_host();
      dec_ids.insert(dec_ids.end(), ids.begin(), ids.end());
      dec_w.insert(dec_w.end(), w.begin(), w.end());
    }

    for (const unsigned reps : {1u, 24u}) {
      CAPTURE(reps);
      const unsigned n = n_prompt * reps;
      std::vector<float> xr;
      for (unsigned r = 0; r < reps; ++r)
        xr.insert(xr.end(), x.begin(), x.end());
      miso::MoePrefillScratch ps(n);
      miso::DeviceBuffer<float> px(xr), py(xr.size());
      miso::moe_prefill(moe, ps, px.data(), py.data(), n, nullptr);
      MISO_HIP_CHECK(hipDeviceSynchronize());
      const auto ids = ps.ids.to_host();
      const auto w = ps.weights.to_host();
      const auto y = py.to_host();
      double worst = 0;
      for (unsigned t = 0; t < n; ++t) {
        CAPTURE(t);
        const unsigned tp = t % n_prompt;
        for (unsigned k = 0; k < 8; ++k) {
          CHECK(ids[t * 9 + k] == dec_ids[tp * 8 + k]);
          CHECK(ids[t * 9 + k] == static_cast<int>(want_id[tp * 8 + k]));
          CHECK(w[t * 9 + k] == dec_w[tp * 8 + k]);
        }
        double num = 0, den = 0;
        for (unsigned i = 0; i < kHidden; ++i) {
          const double d = double{y[t * kHidden + i]} - want[tp * kHidden + i];
          num += d * d, den += double{want[tp * kHidden + i]} * want[tp * kHidden + i];
        }
        const double rel = std::sqrt(num / den);
        worst = std::max(worst, rel);
        CHECK(rel <= 1e-3);
        if (t >= n_prompt)
          CHECK(std::memcmp(&y[t * kHidden], &y[tp * kHidden], kHidden * 4) == 0);
      }
      MESSAGE("layer " << layer << " x" << reps << ": max relative L2 error " << worst);
    }
  }
}
