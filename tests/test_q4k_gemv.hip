#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "dequant.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "q4k_gemv.hpp"
#include "repack.hpp"

using miso::gguf::File;
using miso::kernels::ActFormat;

namespace {

constexpr int kBlock = 256;

struct Case {
  std::vector<std::uint8_t> w;  // Q4_K r1 rows (what the kernel consumes)
  std::vector<float> wf;        // CPU-dequantised rows (from the original GGUF bytes)
  std::vector<float> ds;        // d * sc of each element's sub-block
  unsigned n = 0, k = 0;
};

Case load(const File& f, const std::string& name) {
  const auto* t = f.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  REQUIRE(t->type == miso::gguf::GgmlType::Q4_K);
  Case c;
  c.k = static_cast<unsigned>(t->dims[0]);
  c.n = static_cast<unsigned>(t->n_elements() / c.k);
  const auto bytes = f.tensor_data(*t);
  c.wf.resize(t->n_elements());
  miso::dequantize(t->type, bytes, c.wf);
  std::vector<std::byte> r1(bytes.size());
  miso::repack_q4k(bytes, r1);
  c.w.resize(r1.size());
  std::memcpy(c.w.data(), r1.data(), r1.size());
  c.ds.resize(t->n_elements());
  for (std::size_t b = 0; b < bytes.size() / miso::kQ4kBlockBytes; ++b) {
    const std::byte* blk = bytes.data() + b * miso::kQ4kBlockBytes;
    std::uint16_t dbits;
    std::memcpy(&dbits, blk, 2);
    for (int j = 0; j < 8; ++j) {
      std::uint8_t sc, m;
      miso::q4k_scale_min(blk, j, sc, m);
      std::fill_n(&c.ds[b * 256 + 32 * j], 32, miso::fp16_to_f32(dbits) * sc);
    }
  }
  return c;
}

// Per-element activation error of each format, used for a rigorous per-row bound.
std::vector<double> act_error(ActFormat a, const std::vector<float>& x) {
  std::vector<double> e(x.size());
  for (std::size_t b = 0; b < x.size(); b += 32) {
    float amax = 0;
    for (std::size_t i = b; i < b + 32; ++i)
      amax = std::max(amax, std::fabs(x[i]));
    for (std::size_t i = b; i < b + 32; ++i) {
      e[i] = a == ActFormat::Int8 ? 0.5 * amax / 127.0 : std::fabs(x[i]) * 0x1p-11 + 0x1p-25;
    }
  }
  return e;
}

template <ActFormat A, unsigned K, int R>
std::vector<float> run_gpu(const Case& c, const std::vector<float>& x, unsigned grid) {
  miso::DeviceBuffer<std::uint8_t> dw(c.w);
  miso::DeviceBuffer<float> dx(x), dy(std::vector<float>(c.n, -1.0f));
  miso::kernels::Q4kGemvParams p{dw.data(), dx.data(), dy.data(), c.n};
  miso::kernels::q4k_gemv_kernel<A, K, kBlock, R><<<grid, kBlock>>>(p);
  MISO_HIP_CHECK(hipGetLastError());
  return dy.to_host();
}

// Checks |y - W x| <= sum_k |W_ik| e_k + K * 2^-23 * sum_k |W_ik x_k| (+ FP16 offset term) per row.
// The FP16 variant accumulates (1024 + q) * x in FP32 and subtracts 1024 * sum(x) per 16-element
// lane chain, adding at most 16 * 2^-24 * sum_k (1039 * d * sc_k) |x_k|.
template <ActFormat A, unsigned K, int R = 1>
void check(const Case& c, const std::vector<float>& x, unsigned grid) {
  REQUIRE(c.k == K);
  const auto y = run_gpu<A, K, R>(c, x, grid);
  const auto e = act_error(A, x);
  double worst = 0;
  for (unsigned r = 0; r < c.n; ++r) {
    const float* w = &c.wf[std::size_t{r} * K];
    double ref = 0, mag = 0, bound = 0;
    for (unsigned k = 0; k < K; ++k) {
      ref += double{w[k]} * x[k];
      mag += std::fabs(double{w[k]} * x[k]);
      bound += std::fabs(double{w[k]}) * e[k];
      if (A == ActFormat::Fp16)
        bound += 0x1p-20 * 1039.0 * c.ds[std::size_t{r} * K + k] * std::fabs(x[k]);
    }
    bound += K * 0x1p-23 * mag;
    const double err = std::fabs(y[r] - ref);
    worst = std::max(worst, err / bound);
    if (!(err <= bound)) {
      FAIL_CHECK("row " << r << ": y=" << y[r] << " ref=" << ref << " err=" << err
                        << " bound=" << bound);
      return;
    }
  }
  MESSAGE("max error / bound = " << worst);
}

std::vector<float> random_x(unsigned k) {
  std::mt19937 rng(7);
  std::normal_distribution<float> d(0.0f, 1.5f);
  std::vector<float> x(k);
  for (auto& v : x)
    v = d(rng);
  return x;
}

bool model_present() {
  return std::filesystem::exists(MISO_ORNITH_GGUF);
}

}  // namespace

TEST_CASE("Q4_K GEMV matches the CPU reference within the format error bound (fixture)") {
  const File f = File::open(std::string(MISO_FIXTURE_DIR) + "/tiny.gguf");
  const Case c = load(f, "q4_k");  // K = 8192, N = 2, real model blocks
  const auto x = random_x(c.k);
  SUBCASE("int8 activations") {
    check<ActFormat::Int8, 8192>(c, x, 1);
  }
  SUBCASE("fp16 activations") {
    check<ActFormat::Fp16, 8192>(c, x, 1);
  }
}

TEST_CASE("Q4_K GEMV on real projections with a real activation" *
          doctest::skip(!model_present())) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto* moe_in = g.find_tensor("layer0.moe_in");
  REQUIRE(moe_in != nullptr);
  const auto* xs = reinterpret_cast<const float*>(g.tensor_data(*moe_in).data());
  const std::vector<float> x2048(xs, xs + 2048);  // token 0, post-attention RMSNorm output

  SUBCASE("attn_gate 2048 -> 4096, int8") {
    check<ActFormat::Int8, 2048>(load(m, "blk.0.attn_gate.weight"), x2048, 240);
  }
  SUBCASE("attn_gate 2048 -> 4096, fp16") {
    check<ActFormat::Fp16, 2048>(load(m, "blk.0.attn_gate.weight"), x2048, 240);
  }
  SUBCASE("attn_gate 2048 -> 4096, fp16, 4 rows per wave, grid not dividing N") {
    check<ActFormat::Fp16, 2048, 4>(load(m, "blk.0.attn_gate.weight"), x2048, 37);
  }
  std::vector<float> x4096(x2048);
  x4096.insert(x4096.end(), xs + 2048, xs + 4096);  // two tokens back to back
  SUBCASE("ssm_out 4096 -> 2048, int8") {
    check<ActFormat::Int8, 4096>(load(m, "blk.0.ssm_out.weight"), x4096, 7);
  }
  SUBCASE("ssm_out 4096 -> 2048, fp16") {
    check<ActFormat::Fp16, 4096>(load(m, "blk.0.ssm_out.weight"), x4096, 7);
  }
}
