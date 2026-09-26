#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "embed.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "repack.hpp"
#include "rmsnorm.hpp"

using miso::gguf::File;

namespace {

constexpr unsigned kHidden = 2048;
constexpr int kBlock = 256;

std::vector<float> f32(const File& f, const std::string& name) {
  const auto* t = f.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  REQUIRE(t->type == miso::gguf::GgmlType::F32);
  const auto b = f.tensor_data(*t);
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

bool model_present() {
  return std::filesystem::exists(MISO_ORNITH_GGUF);
}

File golden() {
  return File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
}

}  // namespace

TEST_CASE("add + RMSNorm reproduces the golden post-attention norm" *
          doctest::skip(!model_present())) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = golden();
  const float eps = static_cast<float>(m.get<double>("qwen35moe.attention.layer_norm_rms_epsilon"));
  for (const int layer : {0, 3}) {
    CAPTURE(layer);
    const std::string p = "layer" + std::to_string(layer) + ".";
    const auto in = f32(g, p + "in"), mixer = f32(g, p + "mixer_out"),
               moe_in = f32(g, p + "moe_in");
    const auto w = f32(m, "blk." + std::to_string(layer) + ".post_attention_norm.weight");
    const unsigned n_tok = static_cast<unsigned>(in.size() / kHidden);

    miso::DeviceBuffer<float> res(in), delta(mixer), weight(w), out(in.size());
    miso::kernels::AddRmsNormParams prm{res.data(), delta.data(), weight.data(), out.data(), eps};
    miso::kernels::add_rmsnorm_kernel<kHidden, miso::kernels::kRmsNormBlock>
        <<<5, miso::kernels::kRmsNormBlock>>>(prm, n_tok);
    MISO_HIP_CHECK(hipGetLastError());
    const auto h = res.to_host(), y = out.to_host();

    std::size_t bad_h = 0, bad_y = 0;
    double worst = 0;
    for (std::size_t i = 0; i < in.size(); ++i) {
      bad_h += h[i] != in[i] + mixer[i];
      // Reduction order and rsqrt differ from torch: |dy| <= |y| (n/2 + 8) 2^-24.
      const double bound = std::fabs(moe_in[i]) * (kHidden / 2 + 8) * 0x1p-24 + 1e-30;
      const double e = std::fabs(double{y[i]} - moe_in[i]);
      worst = std::max(worst, e / bound);
      bad_y += !(e <= bound);
    }
    CHECK(bad_h == 0);
    CHECK(bad_y == 0);
    MESSAGE("max error / bound = " << worst);
  }
}

TEST_CASE("Q4_K r1 embedding lookup equals the golden input bit for bit" *
          doctest::skip(!model_present())) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = golden();
  const auto& ids = g.get<std::vector<std::int64_t>>("golden.token_ids");
  const auto emb = f32(g, "embeddings");

  const auto* t = m.find_tensor("token_embd.weight");
  REQUIRE(t != nullptr);
  const auto src = m.tensor_data(*t);
  std::vector<std::byte> r1(src.size());
  miso::repack_q4k(src, r1);
  std::vector<std::uint8_t> r1u(r1.size());
  std::memcpy(r1u.data(), r1.data(), r1.size());
  std::vector<std::int32_t> ids32(ids.begin(), ids.end());

  miso::DeviceBuffer<std::uint8_t> table(r1u);
  miso::DeviceBuffer<std::int32_t> tok(ids32);
  miso::DeviceBuffer<float> out(ids.size() * kHidden);
  miso::kernels::EmbedParams prm{table.data(), tok.data(), out.data()};
  miso::kernels::embed_q4k_kernel<kHidden, kBlock>
      <<<3, kBlock>>>(prm, static_cast<unsigned>(ids.size()));
  MISO_HIP_CHECK(hipGetLastError());
  const auto got = out.to_host();
  CHECK(std::memcmp(got.data(), emb.data(), emb.size() * sizeof(float)) == 0);
}
