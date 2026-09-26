#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "engine/weights.hpp"
#include "hip_util.hpp"
#include "moe_down_reference.hpp"

namespace k = miso::kernels;

namespace {

template <k::QType T>
__global__ void compare_slot_kernel(const float* x, typename k::moe_detail::Down<T>::Act* rounded,
                                    typename k::moe_detail::Down<T>::Act* original) {
  __shared__ _Float16 h[512];
  for (unsigned i = threadIdx.x; i < 512; i += 256)
    h[i] = static_cast<_Float16>(x[i]);
  __syncthreads();
  if (threadIdx.x < 16) {
    rounded[threadIdx.x] = k::moe_detail::Down<T>::act(h, threadIdx.x);
    original[threadIdx.x] = k::moe_detail::Down<T>::act(x, threadIdx.x);
  }
}

std::vector<float> activations(unsigned n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  // Halfway ties, just above/below a tie, FP16 subnormals, signed zero, and cancellation.
  const float special[] = {0.0f,
                           -0.0f,
                           0x1p-25f,
                           -0x1p-25f,
                           0x1p-24f,
                           -0x1p-24f,
                           1.00048828125f,
                           std::nextafter(1.00048828125f, 0.0f),
                           std::nextafter(1.00048828125f, 2.0f),
                           -1.00048828125f,
                           2048.0f,
                           -2048.0f};
  std::vector<float> v(n);
  for (unsigned i = 0; i < n; ++i)
    v[i] = i % 5 == 0 ? special[(i / 5 + seed) % std::size(special)] : dist(rng);
  return v;
}

template <k::QType T>
void check_slots() {
  using Act = typename k::moe_detail::Down<T>::Act;
  static_assert(sizeof(Act) == 72);  // Packed pairs plus rounded sums, no padding.
  for (unsigned seed : {0u, 7u, 13u}) {
    miso::DeviceBuffer<float> x(activations(512, seed));
    miso::DeviceBuffer<Act> got(16), want(16);
    compare_slot_kernel<T><<<1, 256>>>(x.data(), got.data(), want.data());
    MISO_HIP_CHECK(hipGetLastError());
    const auto g = got.to_host(), w = want.to_host();
    CHECK(std::memcmp(g.data(), w.data(), g.size() * sizeof(Act)) == 0);
  }
}

template <k::QType T>
void check_down(const miso::gguf::File& f, int layer) {
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  const auto exps = miso::upload_qmatrix(f, prefix + "ffn_down_exps.weight");
  const auto shared = miso::upload_qmatrix(f, prefix + "ffn_down_shexp.weight");
  const auto expected_type =
      T == k::QType::Q4K ? miso::gguf::GgmlType::Q4_K : miso::gguf::GgmlType::Q6_K;
  REQUIRE(exps.type == expected_type);
  REQUIRE(shared.type == expected_type);
  miso::DeviceBuffer<int> ids(std::vector<int>{255, 0, 131, 63, 200, 1, 127, 42});
  miso::DeviceBuffer<float> weights(
      std::vector<float>{0.4f, 0.2f, 0.15f, 0.1f, 0.08f, 0.04f, 0.03f, 0.0f});
  miso::DeviceBuffer<float> got(2048), want(2048);
  for (unsigned seed : {0u, 7u, 13u}) {
    CAPTURE(layer);
    CAPTURE(seed);
    miso::DeviceBuffer<float> h(activations(9 * 512, seed));
    std::vector<float> logits(257, 0.0f);
    logits[256] = seed == 0 ? -100.0f : (seed == 7 ? 0.0f : 100.0f);
    miso::DeviceBuffer<float> dl(logits);
    k::MoeDownParams p{dl.data(),          ids.data(), weights.data(), exps.data.data(),
                       shared.data.data(), h.data(),   got.data()};
    k::moe_down_kernel<T, 256><<<120, 256>>>(p);
    p.out = want.data();
    k::fp32_down_reference_kernel<T, 256><<<120, 256>>>(p);
    MISO_HIP_CHECK(hipGetLastError());
    const auto g = got.to_host(), w = want.to_host();
    for (unsigned i = 0; i < g.size(); ++i) {
      CAPTURE(i);
      REQUIRE(std::isfinite(g[i]));
      CHECK(std::bit_cast<std::uint32_t>(g[i]) == std::bit_cast<std::uint32_t>(w[i]));
    }
  }
}

}  // namespace

TEST_CASE("FP16 staged Q4 and Q6 slots preserve packed bits and rounded offset sums") {
  check_slots<k::QType::Q4K>();
  check_slots<k::QType::Q6K>();
}

TEST_CASE("MoE down staging is bit-identical to FP32 LDS for both quantization types" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const auto f = miso::gguf::File::open(MISO_ORNITH_GGUF);
  check_down<k::QType::Q4K>(f, 5);
  check_down<k::QType::Q6K>(f, 0);
}
