#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "q4k_gemv.hpp"
#include "q6k_gemv.hpp"

using miso::gguf::File;
using miso::kernels::ActFormat;

namespace {

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(double{a[i]} - b[i]));
  return m;
}

}  // namespace

// Shape-tuned gemv() (rows-per-wave / grid) must match the R = 1 reference bit for bit: each row's
// arithmetic is independent of how many rows a wave owns.
TEST_CASE("engine gemv matches R=1 reference for ssm_out and Q6_K qkv" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  miso::DeviceBuffer<float> x(std::vector<float>(4096, 0.02f));

  {
    const auto w = miso::upload_qmatrix(m, "blk.0.ssm_out.weight");
    REQUIRE(w.k == 4096);
    REQUIRE(w.n == 2048);
    miso::DeviceBuffer<float> y(w.n), yref(w.n);
    miso::gemv(w, x.data(), y.data(), nullptr);
    const miso::kernels::Q4kGemvParams p{w.data.data(), x.data(), yref.data(), w.n};
    miso::kernels::q4k_gemv_kernel<ActFormat::Fp16, 4096, 256, 1><<<240, 256>>>(p);
    MISO_HIP_CHECK(hipDeviceSynchronize());
    CHECK(max_abs(y.to_host(), yref.to_host()) == 0.0);
  }
  {
    const auto w = miso::upload_qmatrix(m, "blk.0.attn_qkv.weight");
    REQUIRE(w.type == miso::gguf::GgmlType::Q6_K);
    miso::DeviceBuffer<float> y(w.n), yref(w.n);
    miso::gemv(w, x.data(), y.data(), nullptr);
    const miso::kernels::Q6kGemvParams p{w.data.data(), x.data(), yref.data(), w.n};
    miso::kernels::q6k_gemv_kernel<2048, 256, 1><<<240, 256>>>(p);
    MISO_HIP_CHECK(hipDeviceSynchronize());
    CHECK(max_abs(y.to_host(), yref.to_host()) == 0.0);
  }
}
