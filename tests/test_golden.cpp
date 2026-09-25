#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "dequant.hpp"
#include "gguf.hpp"

using miso::gguf::File;

namespace {

constexpr std::size_t kHidden = 2048;

std::string golden_path() {
  return std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf";
}

std::span<const float> f32(const File& f, const std::string& name) {
  const auto* t = f.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  REQUIRE(t->type == miso::gguf::GgmlType::F32);
  const auto bytes = f.tensor_data(*t);
  return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float)};
}

}  // namespace

TEST_CASE("golden file carries provenance and per-layer tensors") {
  const File g = File::open(golden_path());
  const auto& ids = g.get<std::vector<std::int64_t>>("golden.token_ids");
  CHECK(ids.size() == 17);
  CHECK(g.get<std::vector<std::int64_t>>("golden.layers") == std::vector<std::int64_t>{0, 3});
  CHECK(g.get<std::string>("golden.model_gguf_sha256") ==
        "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f");

  for (const char* layer : {"layer0", "layer3"}) {
    for (const char* part : {"in", "mixer_out", "moe_in", "moe_out", "out"}) {
      const std::string name = std::string(layer) + "." + part;
      CAPTURE(name);
      const auto* t = g.find_tensor(name);
      REQUIRE(t != nullptr);
      CHECK(t->dims == std::vector<std::uint64_t>{kHidden, ids.size()});
    }
  }
}

TEST_CASE("golden layers follow the pre-norm residual structure out = (in + mixer) + moe") {
  const File g = File::open(golden_path());
  for (const std::string layer : {"layer0", "layer3"}) {
    CAPTURE(layer);
    const auto in = f32(g, layer + ".in");
    const auto mixer = f32(g, layer + ".mixer_out");
    const auto moe = f32(g, layer + ".moe_out");
    const auto out = f32(g, layer + ".out");
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
      const float h = in[i] + mixer[i];
      if (h + moe[i] != out[i])
        ++mismatches;
    }
    CHECK(mismatches == 0);
  }
}

TEST_CASE("CPU-dequantised token embeddings equal the golden reference input" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File g = File::open(golden_path());
  const File m = File::open(MISO_ORNITH_GGUF);
  const auto& ids = g.get<std::vector<std::int64_t>>("golden.token_ids");
  const auto emb = f32(g, "embeddings");

  const auto* t = m.find_tensor("token_embd.weight");
  REQUIRE(t != nullptr);
  const std::size_t row_bytes = t->n_bytes() / t->dims[1];
  std::vector<float> row(kHidden);
  for (std::size_t r = 0; r < ids.size(); ++r) {
    CAPTURE(ids[r]);
    miso::dequantize(t->type, m.tensor_data(*t).subspan(ids[r] * row_bytes, row_bytes), row);
    CHECK(std::memcmp(row.data(), emb.data() + r * kHidden, kHidden * sizeof(float)) == 0);
  }
}
