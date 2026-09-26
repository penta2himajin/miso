#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "hip_util.hpp"
#include "q4k_gemv.hpp"
#include "q6k_gemv.hpp"
#include "repack.hpp"

// Bit-exact: GEMV from pre-rounded FP16 activations must match the float load_slot path.

using miso::gguf::File;
using miso::kernels::ActFormat;

namespace {

constexpr int kBlock = 256;

std::vector<_Float16> round_act(const std::vector<float>& x) {
  std::vector<_Float16> h(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    h[i] = static_cast<_Float16>(x[i]);
  return h;
}

bool model_present() {
  return std::filesystem::exists(MISO_ORNITH_GGUF);
}

}  // namespace

TEST_CASE("Q4_K GEMV FP16-act path matches float activations bit-exactly") {
  if (!model_present()) {
    MESSAGE("skipping: Ornith GGUF not at " << MISO_ORNITH_GGUF);
    return;
  }
  const auto f = File::open(MISO_ORNITH_GGUF);
  const auto* t = f.find_tensor("blk.0.attn_gate.weight");
  REQUIRE(t != nullptr);
  REQUIRE(t->type == miso::gguf::GgmlType::Q4_K);
  const unsigned k = static_cast<unsigned>(t->dims[0]);
  const unsigned n = std::min(256u, static_cast<unsigned>(t->n_elements() / k));
  REQUIRE(k == 2048);
  const auto bytes = f.tensor_data(*t).first(std::size_t{n} * (k / 256) * miso::kQ4kBlockBytes);
  std::vector<std::byte> r1(bytes.size());
  miso::repack_q4k(bytes, r1);
  std::vector<std::uint8_t> w(r1.size());
  std::memcpy(w.data(), r1.data(), r1.size());

  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> x(k);
  for (float& v : x)
    v = dist(rng);
  // Round-boundary and tiny values.
  x[0] = 0x1p-24f;
  x[1] = -0.0f;
  x[2] = std::nextafter(0.5f, 1.f);
  x[3] = std::nextafter(0.5f, 0.f);
  const auto xh = round_act(x);

  miso::DeviceBuffer<std::uint8_t> dw(w);
  miso::DeviceBuffer<float> dx(x), dy_f(std::vector<float>(n, -1.f)),
      dy_h(std::vector<float>(n, -2.f));
  miso::DeviceBuffer<_Float16> dxh(xh);
  miso::kernels::Q4kGemvParams pf{dw.data(), dx.data(), dy_f.data(), n, nullptr};
  miso::kernels::Q4kGemvParams ph{dw.data(), dx.data(), dy_h.data(), n, dxh.data()};
  miso::kernels::q4k_gemv_kernel<ActFormat::Fp16, 2048, kBlock, 4><<<120, kBlock>>>(pf);
  miso::kernels::q4k_gemv_kernel<ActFormat::Fp16, 2048, kBlock, 4><<<120, kBlock>>>(ph);
  MISO_HIP_CHECK(hipGetLastError());
  const auto yf = dy_f.to_host(), yh = dy_h.to_host();
  for (unsigned i = 0; i < n; ++i) {
    REQUIRE_MESSAGE(yf[i] == yh[i], "row " << i << " float=" << yf[i] << " half=" << yh[i]);
  }
}

TEST_CASE("Q6_K GEMV FP16-act path matches float activations bit-exactly") {
  if (!model_present()) {
    MESSAGE("skipping: Ornith GGUF not at " << MISO_ORNITH_GGUF);
    return;
  }
  const auto f = File::open(MISO_ORNITH_GGUF);
  const auto* t = f.find_tensor("blk.0.attn_qkv.weight");
  REQUIRE(t != nullptr);
  REQUIRE(t->type == miso::gguf::GgmlType::Q6_K);
  const unsigned k = static_cast<unsigned>(t->dims[0]);
  const unsigned n = std::min(256u, static_cast<unsigned>(t->n_elements() / k));
  REQUIRE(k == 2048);
  const std::size_t nb = k / 256;
  const auto bytes = f.tensor_data(*t).first(std::size_t{n} * nb * miso::kQ6kBlockBytes);
  std::vector<std::byte> r1(std::size_t{n} * miso::q6k_r1_row_bytes(k));
  miso::repack_q6k(bytes, k, r1);
  std::vector<std::uint8_t> w(r1.size());
  std::memcpy(w.data(), r1.data(), r1.size());

  std::mt19937 rng(11);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> x(k);
  for (float& v : x)
    v = dist(rng);
  x[0] = 0x1p-24f;
  x[1] = -0.0f;
  const auto xh = round_act(x);

  miso::DeviceBuffer<std::uint8_t> dw(w);
  miso::DeviceBuffer<float> dx(x), dy_f(std::vector<float>(n, -1.f)),
      dy_h(std::vector<float>(n, -2.f));
  miso::DeviceBuffer<_Float16> dxh(xh);
  miso::kernels::Q6kGemvParams pf{dw.data(), dx.data(), dy_f.data(), n, nullptr};
  miso::kernels::Q6kGemvParams ph{dw.data(), dx.data(), dy_h.data(), n, dxh.data()};
  miso::kernels::q6k_gemv_kernel<2048, kBlock, 2><<<480, kBlock>>>(pf);
  miso::kernels::q6k_gemv_kernel<2048, kBlock, 2><<<480, kBlock>>>(ph);
  MISO_HIP_CHECK(hipGetLastError());
  const auto yf = dy_f.to_host(), yh = dy_h.to_host();
  for (unsigned i = 0; i < n; ++i) {
    REQUIRE_MESSAGE(yf[i] == yh[i], "row " << i << " float=" << yf[i] << " half=" << yh[i]);
  }
}
