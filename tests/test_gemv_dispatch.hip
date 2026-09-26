#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "mixed_gemv.hpp"
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

// Independent mixed-type projections may share one grid while every output keeps the original
// per-row arithmetic. Odd row counts exercise the final partial row group; canaries catch writes
// from a workgroup that incorrectly uses the combined grid's block index or stride.
TEST_CASE("mixed Q6/Q4 GEMV grid matches separate kernels bit for bit" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File model = File::open(MISO_ORNITH_GGUF);
  std::vector<float> values(2048);
  for (unsigned i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.013f;
  values[0] = 1.0f + 0x1p-11f;
  values[1] = -1.0f - 0x1p-11f;
  values[2] = 0x1p-24f;
  values[3] = -0.0f;
  miso::DeviceBuffer<float> x(values);

  const auto check = [&]<int Q6Rows, unsigned Q6Grid, unsigned Q4Grid>(const miso::QMatrix& q6,
                                                                       const miso::QMatrix& q4) {
    hipStream_t explicit_stream = nullptr;
    MISO_HIP_CHECK(hipStreamCreateWithFlags(&explicit_stream, hipStreamNonBlocking));
    for (const auto stream : {static_cast<hipStream_t>(nullptr), explicit_stream}) {
      for (const unsigned trim : {0u, 1u}) {
        const unsigned n6 = q6.n - trim, n4 = q4.n - trim;
        constexpr float kCanary = -8192.25f;
        miso::DeviceBuffer<float> y6(std::vector<float>(n6 + 1, kCanary));
        miso::DeviceBuffer<float> y4(std::vector<float>(n4 + 1, kCanary));
        miso::DeviceBuffer<float> ref6(std::vector<float>(n6 + 1, kCanary));
        miso::DeviceBuffer<float> ref4(std::vector<float>(n4 + 1, kCanary));
        const miso::kernels::Q6kGemvParams p6{q6.data.data(), x.data(), y6.data(), n6};
        const miso::kernels::Q4kGemvParams p4{q4.data.data(), x.data(), y4.data(), n4};
        auto r6 = p6;
        auto r4 = p4;
        r6.y = ref6.data();
        r4.y = ref4.data();
        miso::kernels::q6k_gemv_kernel<2048, 256, Q6Rows><<<Q6Grid, 256, 0, stream>>>(r6);
        miso::kernels::q4k_gemv_kernel<ActFormat::Fp16, 2048, 256, 4>
            <<<Q4Grid, 256, 0, stream>>>(r4);
        miso::kernels::mixed_gemv_kernel<Q6Rows, Q6Grid, Q4Grid>
            <<<Q6Grid + Q4Grid, 256, 0, stream>>>(p6, p4);
        MISO_HIP_CHECK(hipGetLastError());
        MISO_HIP_CHECK(hipStreamSynchronize(stream));
        const auto compare = [](const auto& actual, const auto& expected) {
          REQUIRE(actual.size() == expected.size());
          for (unsigned i = 0; i < actual.size(); ++i)
            CHECK(std::bit_cast<std::uint32_t>(actual[i]) ==
                  std::bit_cast<std::uint32_t>(expected[i]));
        };
        compare(y6.to_host(), ref6.to_host());
        compare(y4.to_host(), ref4.to_host());
        if (trim == 0) {
          for (const bool reverse : {false, true}) {
            miso::DeviceBuffer<float> host6(std::vector<float>(n6 + 1, kCanary));
            miso::DeviceBuffer<float> host4(std::vector<float>(n4 + 1, kCanary));
            if (reverse)
              miso::gemv_pair(q4, q6, x.data(), host4.data(), host6.data(), stream);
            else
              miso::gemv_pair(q6, q4, x.data(), host6.data(), host4.data(), stream);
            MISO_HIP_CHECK(hipStreamSynchronize(stream));
            compare(host6.to_host(), ref6.to_host());
            compare(host4.to_host(), ref4.to_host());
          }
        }
      }
    }
    MISO_HIP_CHECK(hipStreamDestroy(explicit_stream));
  };

  const auto qkv = miso::upload_qmatrix(model, "blk.0.attn_qkv.weight");
  const auto zab = miso::concat_rows(
      model, {"blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"});
  REQUIRE(qkv.n == 8192);
  REQUIRE(zab.n == 4160);
  check.template operator()<2, 480, 240>(qkv, zab);
  const auto v = miso::upload_qmatrix(model, "blk.3.attn_v.weight");
  const auto qk = miso::concat_rows(model, {"blk.3.attn_q.weight", "blk.3.attn_k.weight"});
  REQUIRE(v.n == 512);
  REQUIRE(qk.n == 8704);
  check.template operator()<1, 240, 240>(v, qk);
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
