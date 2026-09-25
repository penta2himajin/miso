#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/deltanet.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

using miso::gguf::File;

namespace {

constexpr unsigned kHidden = 2048;

std::vector<float> golden_f32(const File& g, const std::string& name) {
  const auto* t = g.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  const auto b = g.tensor_data(*t);
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

}  // namespace

// Decoding the prompt one token at a time must reproduce the reference's (chunked, prefill-form)
// DeltaNet output. Tolerance: relative L2 error per token <= 1e-3. An FP64 run of the reference
// with only the GEMV inputs rounded to FP16 (ADR_004) deviates by at most 2.96e-4 on this prompt,
// which is what the engine measures; bugs in head mapping, conv or state give O(1) errors.
TEST_CASE("Gated DeltaNet decode reproduces golden layer 0 token by token" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto in = golden_f32(g, "layer0.in"), want = golden_f32(g, "layer0.mixer_out");
  const unsigned n_tok = static_cast<unsigned>(in.size() / kHidden);

  const auto layer = miso::DeltaNetLayer::load(m, 0);
  miso::DeltaNetState state;
  miso::DeltaNetScratch scratch;
  const float eps = static_cast<float>(m.get<double>("qwen35moe.attention.layer_norm_rms_epsilon"));

  miso::DeviceBuffer<float> h(kHidden), y(kHidden);
  double worst = 0;
  for (unsigned t = 0; t < n_tok; ++t) {
    CAPTURE(t);
    MISO_HIP_CHECK(
        hipMemcpy(h.data(), in.data() + t * kHidden, kHidden * 4, hipMemcpyHostToDevice));
    miso::deltanet_decode(layer, state, scratch, h.data(), nullptr, y.data(), eps, nullptr);
    MISO_HIP_CHECK(hipDeviceSynchronize());
    const auto got = y.to_host();
    double num = 0, den = 0;
    for (unsigned i = 0; i < kHidden; ++i) {
      const double d = double{got[i]} - want[t * kHidden + i];
      num += d * d, den += double{want[t * kHidden + i]} * want[t * kHidden + i];
    }
    const double rel = std::sqrt(num / den);
    worst = std::max(worst, rel);
    CHECK(rel <= 1e-3);
  }
  MESSAGE("max relative L2 error over " << n_tok << " tokens = " << worst);
}
