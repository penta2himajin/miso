#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "attention_decode.hpp"
#include "hip_util.hpp"

namespace k = miso::kernels;
using k::attn::half_t;

namespace {

constexpr unsigned kQ = 16, kKv = 2, kD = 256;

// Split-K decode attention over a random FP16 cache vs an FP64 CPU reference of
// softmax(q K^T / 16) V * sigmoid(gate). All arithmetic after the FP16 loads is FP32, so the
// relative error per head stays within a few ulps of the accumulation length.
void check(unsigned len, unsigned max_splits) {
  CAPTURE(len);
  CAPTURE(max_splits);
  const unsigned max_ctx = len + 7;
  std::mt19937 rng(len);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> q(kQ * kD), qg(kQ * 2 * kD);
  std::vector<std::uint16_t> kc(kKv * max_ctx * kD), vc(kKv * max_ctx * kD);
  std::vector<float> kf(kc.size()), vf(vc.size());
  for (auto& x : q)
    x = 2.0f * nd(rng);
  for (auto& x : qg)
    x = nd(rng);
  for (std::size_t i = 0; i < kc.size(); ++i) {
    const half_t a = static_cast<half_t>(nd(rng)), b = static_cast<half_t>(nd(rng));
    kc[i] = __builtin_bit_cast(std::uint16_t, a), vc[i] = __builtin_bit_cast(std::uint16_t, b);
    kf[i] = static_cast<float>(a), vf[i] = static_cast<float>(b);
  }

  const unsigned n_split = k::attn_n_splits(len, max_splits);
  miso::DeviceBuffer<float> dq(q), dqg(qg), core(kQ * kD);
  miso::DeviceBuffer<std::uint16_t> dk(kc), dv(vc);
  miso::DeviceBuffer<float> pm(kKv * n_split * 8), pl(kKv * n_split * 8),
      po(kKv * n_split * 8 * kD);
  const k::AttnSplitParams sp{dq.data(),
                              reinterpret_cast<const half_t*>(dk.data()),
                              reinterpret_cast<const half_t*>(dv.data()),
                              pm.data(),
                              pl.data(),
                              po.data(),
                              len,
                              max_ctx,
                              n_split};
  k::attn_split_kernel<256><<<dim3(n_split, kKv), 256>>>(sp);
  const k::AttnCombineParams cp{pm.data(), pl.data(), po.data(), dqg.data(), core.data(), n_split};
  k::attn_combine_kernel<256><<<kQ, 256>>>(cp);
  MISO_HIP_CHECK(hipGetLastError());
  const auto got = core.to_host();

  double worst = 0;
  for (unsigned h = 0; h < kQ; ++h) {
    const unsigned kvh = h / (kQ / kKv);
    std::vector<double> s(len);
    double mx = -1e300;
    for (unsigned j = 0; j < len; ++j) {
      double d = 0;
      for (unsigned e = 0; e < kD; ++e)
        d += double{q[h * kD + e]} * kf[(kvh * max_ctx + j) * kD + e];
      s[j] = d / 16.0, mx = std::max(mx, s[j]);
    }
    double l = 0;
    for (auto& x : s)
      x = std::exp(x - mx), l += x;
    double num = 0, den = 0;
    for (unsigned e = 0; e < kD; ++e) {
      double o = 0;
      for (unsigned j = 0; j < len; ++j)
        o += s[j] * vf[(kvh * max_ctx + j) * kD + e];
      const double ref = o / l / (1.0 + std::exp(-double{qg[h * 2 * kD + kD + e]}));
      num += (got[h * kD + e] - ref) * (got[h * kD + e] - ref), den += ref * ref;
    }
    worst = std::max(worst, std::sqrt(num / den));
  }
  CHECK(worst <= 1e-5);
  MESSAGE("len " << len << " splits " << n_split << ": max relative L2 error " << worst);
}

}  // namespace

TEST_CASE("split-K decode attention matches an FP64 reference across chunk boundaries") {
  for (unsigned len : {1u, 31u, 32u, 33u, 257u, 1000u})
    check(len, 120);
}

TEST_CASE("split-K decode attention with long contexts and few splits") {
  check(5000, 120);
  check(3872, 120);  // 121 sub-chunks: 61 splits of 2, none empty
  check(5000, 3);    // many 32-position sub-chunks per workgroup
  CHECK(miso::kernels::attn_n_splits(3872, 120) == 61);
  CHECK(miso::kernels::attn_n_splits(33, 120) == 2);
}
