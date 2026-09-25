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
#include "q6k_gemv.hpp"
#include "repack.hpp"

using miso::gguf::File;

namespace {

constexpr int kBlock = 256;

struct Case {
  std::vector<std::uint8_t> w;  // Q6_K r1 rows
  std::vector<float> wf;        // CPU-dequantised rows
  std::vector<float> ds;        // |d * sc| of each element's 16-group
  unsigned n = 0, k = 0;
};

// Rows [0, max_rows) of a Q6_K tensor.
Case load(const File& f, const std::string& name, unsigned max_rows = ~0u) {
  const auto* t = f.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  REQUIRE(t->type == miso::gguf::GgmlType::Q6_K);
  Case c;
  c.k = static_cast<unsigned>(t->dims[0]);
  c.n = std::min(static_cast<unsigned>(t->n_elements() / c.k), max_rows);
  const std::size_t nb = c.k / 256;
  const auto bytes = f.tensor_data(*t).first(std::size_t{c.n} * nb * miso::kQ6kBlockBytes);
  c.wf.resize(std::size_t{c.n} * c.k);
  miso::dequantize(t->type, bytes, c.wf);
  std::vector<std::byte> r1(std::size_t{c.n} * miso::q6k_r1_row_bytes(c.k));
  miso::repack_q6k(bytes, c.k, r1);
  c.w.resize(r1.size());
  std::memcpy(c.w.data(), r1.data(), r1.size());
  c.ds.resize(c.wf.size());
  for (std::size_t b = 0; b < bytes.size() / miso::kQ6kBlockBytes; ++b) {
    const std::byte* blk = bytes.data() + b * miso::kQ6kBlockBytes;
    std::uint16_t dbits;
    std::memcpy(&dbits, blk + 208, 2);
    for (int g = 0; g < 16; ++g) {
      const auto sc = static_cast<std::int8_t>(blk[192 + g]);
      std::fill_n(&c.ds[b * 256 + 16 * g], 16, std::fabs(miso::fp16_to_f32(dbits) * sc));
    }
  }
  return c;
}

// |y - W x| <= sum |W| e + K 2^-23 sum |W x| + 2^-20 sum 1087 |d sc| |x| for every row, where e is
// the FP16 activation rounding and the last term covers the (1024 + q6) offset cancellation.
template <unsigned K, int R = 1>
void check(const Case& c, const std::vector<float>& x, unsigned grid) {
  REQUIRE(c.k == K);
  miso::DeviceBuffer<std::uint8_t> dw(c.w);
  miso::DeviceBuffer<float> dx(x), dy(std::vector<float>(c.n, -1.0f));
  miso::kernels::Q6kGemvParams p{dw.data(), dx.data(), dy.data(), c.n};
  miso::kernels::q6k_gemv_kernel<K, kBlock, R><<<grid, kBlock>>>(p);
  MISO_HIP_CHECK(hipGetLastError());
  const auto y = dy.to_host();
  double worst = 0;
  for (unsigned r = 0; r < c.n; ++r) {
    double ref = 0, mag = 0, bound = 0;
    for (unsigned k = 0; k < K; ++k) {
      const std::size_t i = std::size_t{r} * K + k;
      ref += double{c.wf[i]} * x[k];
      mag += std::fabs(double{c.wf[i]} * x[k]);
      bound += std::fabs(double{c.wf[i]}) * (std::fabs(x[k]) * 0x1p-11 + 0x1p-25) +
               0x1p-20 * 1087.0 * c.ds[i] * std::fabs(x[k]);
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

bool model_present() {
  return std::filesystem::exists(MISO_ORNITH_GGUF);
}

}  // namespace

TEST_CASE("Q6_K GEMV matches the CPU reference within the FP16 error bound (fixture)") {
  const File f = File::open(std::string(MISO_FIXTURE_DIR) + "/tiny.gguf");
  const Case c = load(f, "q6_k");  // K = 4096, N = 4, real LM-head blocks
  std::mt19937 rng(11);
  std::normal_distribution<float> d(0.0f, 1.5f);
  std::vector<float> x(c.k);
  for (auto& v : x)
    v = d(rng);
  SUBCASE("1 row per wave") {
    check<4096>(c, x, 1);
  }
  SUBCASE("4 rows per wave (tail)") {
    check<4096, 4>(c, x, 1);
  }
}

TEST_CASE("Q6_K GEMV on real projections with a real activation" *
          doctest::skip(!model_present())) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto* xs =
      reinterpret_cast<const float*>(g.tensor_data(*g.find_tensor("layer0.moe_in")).data());
  const std::vector<float> x(xs, xs + 2048);
  SUBCASE("DeltaNet qkv 2048 -> 8192") {
    check<2048>(load(m, "blk.0.attn_qkv.weight"), x, 240);
  }
  SUBCASE("DeltaNet qkv, 4 rows per wave, non-dividing grid") {
    check<2048, 4>(load(m, "blk.0.attn_qkv.weight"), x, 37);
  }
  SUBCASE("LM head, first 8192 rows") {
    check<2048>(load(m, "output.weight", 8192), x, 120);
  }
}
