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
#include "qgemm.hpp"
#include "repack.hpp"

using miso::gguf::File;
using miso::kernels::QFormat;

namespace {

struct Case {
  QFormat format;
  std::vector<std::uint8_t> w;  // r1 rows (what the kernel consumes)
  std::vector<float> wf;        // CPU-dequantised rows
  std::vector<float> mag;       // per-element upper bound of |weight term| magnitudes (see check)
  unsigned n = 0, k = 0;
};

// Rows [0, max_rows) of a Q4_K or Q6_K tensor.
Case load(const File& f, const std::string& name, unsigned max_rows = ~0u) {
  const auto* t = f.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  Case c;
  c.k = static_cast<unsigned>(t->dims[0]);
  c.n = std::min(static_cast<unsigned>(t->n_elements() / c.k), max_rows);
  const std::size_t nb = c.k / 256, n_blocks = std::size_t{c.n} * nb;
  c.wf.resize(std::size_t{c.n} * c.k);
  c.mag.resize(c.wf.size());
  std::vector<std::byte> r1;
  if (t->type == miso::gguf::GgmlType::Q4_K) {
    c.format = QFormat::Q4K;
    const auto bytes = f.tensor_data(*t).first(n_blocks * miso::kQ4kBlockBytes);
    miso::dequantize(t->type, bytes, c.wf);
    r1.resize(bytes.size());
    miso::repack_q4k(bytes, r1);
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const std::byte* blk = bytes.data() + b * miso::kQ4kBlockBytes;
      std::uint16_t dbits[2];
      std::memcpy(dbits, blk, 4);
      for (int j = 0; j < 8; ++j) {
        std::uint8_t sc, m;
        miso::q4k_scale_min(blk, j, sc, m);
        const double a = 15.0 * miso::fp16_to_f32(dbits[0]) * sc + miso::fp16_to_f32(dbits[1]) * m;
        std::fill_n(&c.mag[b * 256 + 32 * j], 32, static_cast<float>(a));
      }
    }
  } else {
    REQUIRE(t->type == miso::gguf::GgmlType::Q6_K);
    c.format = QFormat::Q6K;
    const auto bytes = f.tensor_data(*t).first(n_blocks * miso::kQ6kBlockBytes);
    miso::dequantize(t->type, bytes, c.wf);
    r1.resize(std::size_t{c.n} * miso::q6k_r1_row_bytes(c.k));
    miso::repack_q6k(bytes, c.k, r1);
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const std::byte* blk = bytes.data() + b * miso::kQ6kBlockBytes;
      std::uint16_t dbits;
      std::memcpy(&dbits, blk + 208, 2);
      for (int g = 0; g < 16; ++g) {
        const auto sc = static_cast<std::int8_t>(blk[192 + g]);
        std::fill_n(&c.mag[b * 256 + 16 * g], 16,
                    std::fabs(32.0f * miso::fp16_to_f32(dbits) * static_cast<float>(sc)));
      }
    }
  }
  c.w.resize(r1.size());
  std::memcpy(c.w.data(), r1.data(), r1.size());
  return c;
}

struct Act {
  std::vector<_Float16> xh;
  std::vector<float> xs;
};

// FP16 rounding and per-32 sums exactly as qgemm_act_op computes them.
Act cpu_act(const std::vector<float>& x) {
  Act a;
  a.xh.resize(x.size());
  a.xs.resize(x.size() / 32);
  for (std::size_t g = 0; g < a.xs.size(); ++g) {
    float s = 0;
    for (int c = 0; c < 8; ++c) {
      float q[4];
      for (int i = 0; i < 4; ++i) {
        a.xh[32 * g + 4 * c + i] = static_cast<_Float16>(x[32 * g + 4 * c + i]);
        q[i] = static_cast<float>(a.xh[32 * g + 4 * c + i]);
      }
      s += q[0] + q[1] + q[2] + q[3];
    }
    a.xs[g] = s;
  }
  return a;
}

struct GpuAct {
  miso::DeviceBuffer<_Float16> xh;
  miso::DeviceBuffer<float> xs;
};

GpuAct gpu_act(const std::vector<float>& x, unsigned k) {
  miso::DeviceBuffer<float> dx(x);
  GpuAct a{miso::DeviceBuffer<_Float16>(x.size()), miso::DeviceBuffer<float>(x.size() / 32)};
  const miso::kernels::QGemmActParams p{dx.data(), a.xh.data(), a.xs.data(), k,
                                        static_cast<unsigned>(x.size() / k)};
  miso::kernels::qgemm_act_kernel<<<64, 256>>>(p);
  MISO_HIP_CHECK(hipGetLastError());
  MISO_HIP_CHECK(hipDeviceSynchronize());
  return a;
}

std::vector<float> run_gpu(const Case& c, const GpuAct& a, unsigned n_tok, unsigned grid) {
  miso::DeviceBuffer<std::uint8_t> dw(c.w);
  const unsigned ldy = c.n + 3;  // exercises a row stride different from N
  miso::DeviceBuffer<float> dy(std::vector<float>(std::size_t{n_tok} * ldy, -1.0f));
  const miso::kernels::QGemmParams p{dw.data(), a.xh.data(), a.xs.data(), dy.data(),
                                     c.k,       c.n,         n_tok,       ldy};
  if (c.format == QFormat::Q4K)
    miso::kernels::qgemm_kernel<QFormat::Q4K><<<grid, 256>>>(p);
  else
    miso::kernels::qgemm_kernel<QFormat::Q6K><<<grid, 256>>>(p);
  MISO_HIP_CHECK(hipGetLastError());
  auto y = dy.to_host();
  for (unsigned t = 0; t < n_tok; ++t)
    for (unsigned n = c.n; n < ldy; ++n)
      REQUIRE_MESSAGE(y[std::size_t{t} * ldy + n] == -1.0f, "write past N at token " << t);
  std::vector<float> out(std::size_t{n_tok} * c.n);
  for (unsigned t = 0; t < n_tok; ++t)
    std::copy_n(&y[std::size_t{t} * ldy], c.n, &out[std::size_t{t} * c.n]);
  return out;
}

// Reference: the same FP16 activations in FP64 against the exact weights. The kernel only adds
// FP32 accumulation error: fdot2 chains of <= 64 products per stage, per-stage scaling and the
// Q4_K min term (whose sums carry 32 roundings), and K / 16 stage or group additions. With
// |term_k| <= mag_k |h_k| (Q4_K: 15 d sc + dmin m; Q6_K: 32 |d sc|), the error is below
// (K / 16 + 80) 2^-24 sum_k mag_k |h_k|.
void check(const Case& c, const std::vector<float>& x, unsigned grid = 0) {
  const unsigned n_tok = static_cast<unsigned>(x.size() / c.k);
  if (grid == 0)
    grid = miso::kernels::qgemm_tiles(c.n, n_tok);
  const Act a = cpu_act(x);
  const GpuAct ga = gpu_act(x, c.k);
  const auto y = run_gpu(c, ga, n_tok, grid);
  const double factor = (c.k / 16.0 + 80.0) * 0x1p-24;
  double worst = 0;
  for (unsigned t = 0; t < n_tok; ++t) {
    const _Float16* h = &a.xh[std::size_t{t} * c.k];
    for (unsigned r = 0; r < c.n; ++r) {
      const float* w = &c.wf[std::size_t{r} * c.k];
      const float* m = &c.mag[std::size_t{r} * c.k];
      double ref = 0, bound = 0;
      for (unsigned k = 0; k < c.k; ++k) {
        const double hk = static_cast<double>(h[k]);
        ref += double{w[k]} * hk;
        bound += double{m[k]} * std::fabs(hk);
      }
      bound = bound * factor + 1e-30;
      const double err = std::fabs(y[std::size_t{t} * c.n + r] - ref);
      worst = std::max(worst, err / bound);
      if (!(err <= bound)) {
        FAIL_CHECK("token " << t << " row " << r << ": y=" << y[std::size_t{t} * c.n + r]
                            << " ref=" << ref << " err=" << err << " bound=" << bound);
        return;
      }
    }
  }
  MESSAGE("max error / bound = " << worst);
}

std::vector<float> random_x(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 1.5f);
  std::vector<float> x(n);
  for (auto& v : x)
    v = d(rng);
  return x;
}

bool model_present() {
  return std::filesystem::exists(MISO_ORNITH_GGUF);
}

}  // namespace

TEST_CASE("GEMM activation prep rounds to FP16 and sums 32-groups like the CPU reference") {
  const auto x = random_x(70 * 2048, 3);
  const Act want = cpu_act(x);
  const GpuAct got = gpu_act(x, 2048);
  const auto xh = got.xh.to_host();
  const auto xs = got.xs.to_host();
  CHECK(std::memcmp(xh.data(), want.xh.data(), xh.size() * sizeof(_Float16)) == 0);
  CHECK(std::memcmp(xs.data(), want.xs.data(), xs.size() * sizeof(float)) == 0);
}

TEST_CASE("GEMM matches the FP64 reference within the FP32 accumulation bound (fixture)") {
  const File f = File::open(std::string(MISO_FIXTURE_DIR) + "/tiny.gguf");
  SUBCASE("Q4_K, K = 8192, N = 2, 70 tokens") {
    const Case c = load(f, "q4_k");
    check(c, random_x(70 * c.k, 5));
  }
  SUBCASE("Q6_K, K = 4096, N = 4, 70 tokens") {
    const Case c = load(f, "q6_k");
    check(c, random_x(70 * c.k, 6));
  }
}

TEST_CASE("GEMM on real projections" * doctest::skip(!model_present())) {
  const File m = File::open(MISO_ORNITH_GGUF);
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto* moe_in = g.find_tensor("layer0.moe_in");
  REQUIRE(moe_in != nullptr);
  const auto* xs = reinterpret_cast<const float*>(g.tensor_data(*moe_in).data());
  const std::vector<float> x17(xs, xs + 17 * 2048);  // 17 post-attention RMSNorm outputs
  SUBCASE("Q4_K attn_gate 2048 -> 4096, 17 real tokens") {
    check(load(m, "blk.0.attn_gate.weight"), x17);
  }
  SUBCASE("Q4_K ssm_alpha 2048 -> 32 (partial row tile), 17 real tokens") {
    check(load(m, "blk.0.ssm_alpha.weight"), x17);
  }
  SUBCASE("Q6_K attn_qkv 2048 -> 8192, 17 real tokens") {
    check(load(m, "blk.0.attn_qkv.weight"), x17);
  }
  SUBCASE("Q4_K ssm_out 4096 -> 2048, 130 tokens, grid smaller than the tile count") {
    check(load(m, "blk.0.ssm_out.weight"), random_x(130 * 4096, 9), 37);
  }
  SUBCASE("Q6_K attn_v 2048 -> 512, 130 tokens") {
    const Case c = load(m, "blk.3.attn_v.weight");
    REQUIRE(c.format == QFormat::Q6K);
    check(c, random_x(130 * 2048, 10));
  }
}
