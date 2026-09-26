#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/moe.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "rmsnorm.hpp"

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
      miso::moe_decode(moe, sc, nullptr, nullptr, nullptr, dx.data(), dy.data(), 0.f, nullptr);
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

// Fused post-attention RMSNorm inside the router must match a separate add_rmsnorm + MoE.
TEST_CASE("MoE decode fused post-norm matches separate RMSNorm + MoE" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto x = golden_f32(g, "layer0.moe_in");
  const auto moe = miso::MoeLayer::load(m, 0);
  const auto* nw_t = m.find_tensor("blk.0.post_attention_norm.weight");
  REQUIRE(nw_t != nullptr);
  std::vector<float> nw(2048);
  std::memcpy(nw.data(), m.tensor_data(*nw_t).data(), 2048 * 4);
  constexpr float kEps = 1e-6f;
  miso::MoeScratch sc;
  miso::DeviceBuffer<float> xn(2048), y_sep(2048), y_fuse(2048), dnw(nw);
  {
    miso::DeviceBuffer<float> res(x);
    const miso::kernels::AddRmsNormParams norm{res.data(), nullptr, dnw.data(), xn.data(), kEps};
    miso::kernels::add_rmsnorm_kernel<2048, 256><<<1, 256>>>(norm, 1);
    miso::moe_decode(moe, sc, nullptr, nullptr, nullptr, xn.data(), y_sep.data(), 0.f, nullptr);
    MISO_HIP_CHECK(hipDeviceSynchronize());
  }
  {
    miso::DeviceBuffer<float> res(x);
    miso::moe_decode(moe, sc, res.data(), nullptr, dnw.data(), nullptr, y_fuse.data(), kEps,
                     nullptr);
    MISO_HIP_CHECK(hipDeviceSynchronize());
  }
  const auto a = y_sep.to_host(), b = y_fuse.to_host();
  double num = 0, den = 0;
  for (unsigned i = 0; i < 2048; ++i) {
    const double d = double{a[i]} - b[i];
    num += d * d;
    den += double{a[i]} * a[i];
  }
  const double rel = std::sqrt(num / (den + 1e-30));
  MESSAGE("fused vs separate relative L2 " << rel);
  CHECK(rel <= 1e-6);
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
      miso::moe_decode(moe, sc, nullptr, nullptr, nullptr, dx.data(), dy.data(), 0.f, nullptr);
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
