#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/attention.hpp"
#include "engine/deltanet.hpp"
#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

using miso::gguf::File;

namespace {

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(double{a[i]} - b[i]));
  return m;
}

}  // namespace

// When attn_qkv is Q4_K (same as gate/alpha/beta), load concatenates all four into one matrix so
// decode issues one GEMV instead of two. The fused output must match separate GEMVs bit for bit.
TEST_CASE("DeltaNet Q4_K qkv fuses with z_ab and matches separate GEMVs" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const auto fused = miso::DeltaNetLayer::load(m, 5);
  REQUIRE(fused.qkv.type == miso::gguf::GgmlType::Q4_K);
  REQUIRE(fused.qkv.n == 12352);
  REQUIRE(fused.z_ab.n == 0);

  const auto qkv = miso::upload_qmatrix(m, "blk.5.attn_qkv.weight");
  const auto zab = miso::concat_rows(
      m, {"blk.5.attn_gate.weight", "blk.5.ssm_alpha.weight", "blk.5.ssm_beta.weight"});
  REQUIRE(qkv.n == 8192);
  REQUIRE(zab.n == 4160);

  miso::DeviceBuffer<float> x(std::vector<float>(2048, 0.02f));
  miso::DeviceBuffer<float> y_fused(12352), y_parts(12352);
  miso::gemv(fused.qkv, x.data(), y_fused.data(), nullptr);
  miso::gemv(qkv, x.data(), y_parts.data(), nullptr);
  miso::gemv(zab, x.data(), y_parts.data() + 8192, nullptr);
  MISO_HIP_CHECK(hipDeviceSynchronize());
  CHECK(max_abs_diff(y_fused.to_host(), y_parts.to_host()) == 0.0);
}

// Layers where attn_q/k/v are all Q4_K concatenate into one projection GEMV.
TEST_CASE("Attention Q4_K q/k/v fuse and match separate GEMVs" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const auto fused = miso::AttentionLayer::load(m, 11);
  REQUIRE(fused.q.type == miso::gguf::GgmlType::Q4_K);
  REQUIRE(fused.q.n == 9216);
  REQUIRE(fused.k.n == 0);
  REQUIRE(fused.v.n == 0);

  const auto q = miso::upload_qmatrix(m, "blk.11.attn_q.weight");
  const auto k = miso::upload_qmatrix(m, "blk.11.attn_k.weight");
  const auto v = miso::upload_qmatrix(m, "blk.11.attn_v.weight");
  REQUIRE(q.n == 8192);
  REQUIRE(k.n == 512);
  REQUIRE(v.n == 512);

  miso::DeviceBuffer<float> x(std::vector<float>(2048, 0.02f));
  miso::DeviceBuffer<float> y_fused(9216), y_parts(9216);
  miso::gemv(fused.q, x.data(), y_fused.data(), nullptr);
  miso::gemv(q, x.data(), y_parts.data(), nullptr);
  miso::gemv(k, x.data(), y_parts.data() + 8192, nullptr);
  miso::gemv(v, x.data(), y_parts.data() + 8704, nullptr);
  MISO_HIP_CHECK(hipDeviceSynchronize());
  CHECK(max_abs_diff(y_fused.to_host(), y_parts.to_host()) == 0.0);
}

// Layer 0 stays split (Q6_K qkv); layer 3 keeps k/v separate (v is Q6_K).
TEST_CASE("mixed-type layers keep projections separate" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const auto dn = miso::DeltaNetLayer::load(m, 0);
  CHECK(dn.qkv.type == miso::gguf::GgmlType::Q6_K);
  CHECK(dn.qkv.n == 8192);
  CHECK(dn.z_ab.n == 4160);

  const auto at = miso::AttentionLayer::load(m, 3);
  CHECK(at.q.n == 8192);
  CHECK(at.k.n == 512);
  CHECK(at.v.n == 512);
}
