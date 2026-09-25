#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/attention.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

using miso::gguf::File;

namespace {

constexpr unsigned kHidden = 2048, kCore = 4096;

std::vector<float> golden_f32(const File& g, const std::string& name) {
  const auto* t = g.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  const auto b = g.tensor_data(*t);
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

double rel_l2(const std::vector<float>& got, const float* want, std::size_t n) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = double{got[i]} - want[i];
    num += d * d, den += double{want[i]} * want[i];
  }
  return std::sqrt(num / den);
}

}  // namespace

// Decoding the prompt one token at a time (KV cache grows to 17) must reproduce the reference's
// full-attention output of layer 3, both before o_proj (`mixer_core`, gate applied) and after it.
// Tolerance 1e-3 relative L2 per token: an FP64 run of the reference with FP16-rounded GEMV inputs
// and an FP16 KV cache (ADR_002 D9, ADR_004) deviates by at most 4.62e-4, which the engine matches.
TEST_CASE("gated full-attention decode reproduces golden layer 3 token by token" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto in = golden_f32(g, "layer3.in");
  const auto core = golden_f32(g, "layer3.mixer_core");
  const auto out = golden_f32(g, "layer3.mixer_out");
  const unsigned n_tok = static_cast<unsigned>(in.size() / kHidden);

  const auto layer = miso::AttentionLayer::load(m, 3);
  miso::AttentionCache cache(/*max_ctx=*/64);
  miso::AttentionScratch scratch;
  const auto rope = miso::RopeConfig::from(m);
  const float eps = static_cast<float>(m.get<double>("qwen35moe.attention.layer_norm_rms_epsilon"));

  miso::DeviceBuffer<float> h(kHidden), y(kHidden);
  double worst_core = 0, worst_out = 0;
  for (unsigned t = 0; t < n_tok; ++t) {
    CAPTURE(t);
    MISO_HIP_CHECK(
        hipMemcpy(h.data(), in.data() + t * kHidden, kHidden * 4, hipMemcpyHostToDevice));
    miso::attention_decode(layer, cache, scratch, rope, h.data(), nullptr, y.data(), eps, nullptr);
    MISO_HIP_CHECK(hipDeviceSynchronize());
    const double rc = rel_l2(scratch.core.to_host(), core.data() + t * kCore, kCore);
    const double ro = rel_l2(y.to_host(), out.data() + t * kHidden, kHidden);
    worst_core = std::max(worst_core, rc), worst_out = std::max(worst_out, ro);
    CHECK(rc <= 1e-3);
    CHECK(ro <= 1e-3);
  }
  CHECK(cache.len == n_tok);
  MESSAGE("max relative L2 error: core " << worst_core << ", out " << worst_out);
}

// The prefill form (projections as GEMMs over the chunk) must meet the same golden bounds, also
// when the prompt arrives in two chunks.
TEST_CASE("gated full-attention prefill reproduces golden layer 3" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto in = golden_f32(g, "layer3.in");
  const auto core = golden_f32(g, "layer3.mixer_core");
  const auto out = golden_f32(g, "layer3.mixer_out");
  const unsigned n_tok = static_cast<unsigned>(in.size() / kHidden);

  const auto layer = miso::AttentionLayer::load(m, 3);
  const auto rope = miso::RopeConfig::from(m);
  const float eps = static_cast<float>(m.get<double>("qwen35moe.attention.layer_norm_rms_epsilon"));
  for (unsigned first_chunk : {n_tok, 10u}) {
    CAPTURE(first_chunk);
    miso::AttentionCache cache(/*max_ctx=*/64);
    miso::AttentionScratch scratch;
    miso::AttentionPrefillScratch ps(n_tok);
    miso::DeviceBuffer<float> h(in), y(in.size());
    std::vector<float> got_core(std::size_t{n_tok} * kCore);
    for (unsigned t0 = 0; t0 < n_tok; t0 += first_chunk) {
      const unsigned c = std::min(first_chunk, n_tok - t0);
      miso::attention_prefill(layer, cache, scratch, ps, rope, h.data() + t0 * kHidden, nullptr,
                              y.data() + t0 * kHidden, c, eps, nullptr);
      MISO_HIP_CHECK(hipDeviceSynchronize());
      const auto chunk_core = ps.core.to_host();
      std::copy_n(chunk_core.data(), std::size_t{c} * kCore, &got_core[std::size_t{t0} * kCore]);
    }
    const auto got_out = y.to_host();
    double worst_core = 0, worst_out = 0;
    for (unsigned t = 0; t < n_tok; ++t) {
      CAPTURE(t);
      const std::vector<float> gc(&got_core[std::size_t{t} * kCore],
                                  &got_core[std::size_t{t + 1} * kCore]);
      const std::vector<float> go(&got_out[std::size_t{t} * kHidden],
                                  &got_out[std::size_t{t + 1} * kHidden]);
      const double rc = rel_l2(gc, core.data() + t * kCore, kCore);
      const double ro = rel_l2(go, out.data() + t * kHidden, kHidden);
      worst_core = std::max(worst_core, rc), worst_out = std::max(worst_out, ro);
      CHECK(rc <= 1e-3);
      CHECK(ro <= 1e-3);
    }
    CHECK(cache.len == n_tok);
    MESSAGE("max relative L2 error: core " << worst_core << ", out " << worst_out);
  }
}
