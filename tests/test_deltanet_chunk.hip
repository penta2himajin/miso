#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "deltanet_chunk.hpp"
#include "hip_util.hpp"

namespace {

constexpr unsigned kNch = 8192, kV = 32, kD = 128;

double rel(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = double{a[i]} - b[i];
    num += d * d, den += double{b[i]} * b[i];
  }
  return den == 0 ? 0 : std::sqrt(num / den);
}

void check(unsigned n) {
  CAPTURE(n);
  std::mt19937 rng(n + 3);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  auto rnd = [&](std::size_t m) {
    std::vector<float> v(m);
    for (auto& x : v)
      x = nd(rng);
    return v;
  };
  const auto qkv = rnd(std::size_t{n} * kNch), z = rnd(std::size_t{n} * kV * kD);
  const auto a = rnd(std::size_t{n} * kV), b = rnd(std::size_t{n} * kV);
  const auto conv_w = rnd(kNch * 4), conv_in = rnd(kNch * 3);
  auto state = rnd(kV * kD * kD);
  std::vector<float> ssm_a(kV), dt(kV), norm_w(kD);
  for (unsigned h = 0; h < kV; ++h)
    ssm_a[h] = -0.05f * (h + 1), dt[h] = 0.1f * h;
  for (unsigned i = 0; i < kD; ++i)
    norm_w[i] = 1.0f + 0.01f * (int(i) - 64);

  miso::DeviceBuffer<float> dqkv(qkv), dz(z), da(a), db(b), dcw(conv_w), dci(conv_in), dsa(ssm_a),
      ddt(dt), dnw(norm_w);
  miso::DeviceBuffer<float> st_seq(state), st_ch(state), co_seq(kNch * 3), co_ch(kNch * 3);
  miso::DeviceBuffer<float> o_seq(std::size_t{n} * kV * kD), o_ch(std::size_t{n} * kV * kD);
  // z, a and b now share one GEMV output with stride 4160: rows [0,4096) z, [4096,4128) a,
  // [4128,4160) b. Pack the separately generated arrays into that layout.
  std::vector<float> zab(std::size_t{n} * 4160);
  for (unsigned t = 0; t < n; ++t) {
    std::copy_n(z.data() + std::size_t{t} * kV * kD, kV * kD, zab.data() + std::size_t{t} * 4160);
    std::copy_n(a.data() + std::size_t{t} * kV, kV, zab.data() + std::size_t{t} * 4160 + 4096);
    std::copy_n(b.data() + std::size_t{t} * kV, kV, zab.data() + std::size_t{t} * 4160 + 4128);
  }
  miso::DeviceBuffer<float> dzab(zab);
  const miso::kernels::DeltaNetSeqParams seq{dqkv.data(),
                                             dzab.data(),
                                             dzab.data() + 4096,
                                             dzab.data() + 4128,
                                             dcw.data(),
                                             dci.data(),
                                             co_seq.data(),
                                             dsa.data(),
                                             ddt.data(),
                                             dnw.data(),
                                             st_seq.data(),
                                             o_seq.data(),
                                             n,
                                             1e-6f,
                                             4160};
  miso::kernels::deltanet_seq_kernel<256><<<32, 256>>>(seq);
  MISO_HIP_CHECK(hipGetLastError());

  miso::DeviceBuffer<float> q(std::size_t{n} * 16 * kD), k(std::size_t{n} * 16 * kD),
      v(std::size_t{n} * kV * kD), beta(std::size_t{n} * kV), log_a(std::size_t{n} * kV);
  miso::DeviceBuffer<float> sk(kV * 64 * kD), sq(kV * 64 * kD);
  const miso::kernels::DeltaNetConvParams conv{
      dqkv.data(), da.data(), db.data(), dcw.data(), dci.data(),  co_ch.data(), dsa.data(),
      ddt.data(),  q.data(),  k.data(),  v.data(),   beta.data(), log_a.data(), n};
  miso::kernels::deltanet_conv_kernel<<<32, 256>>>(conv);
  miso::kernels::deltanet_norm_kernel<<<64, 256>>>(conv);
  const miso::kernels::DeltaNetChunkParams chunk{q.data(),     k.data(),  v.data(),   beta.data(),
                                                 log_a.data(), dz.data(), dnw.data(), st_ch.data(),
                                                 o_ch.data(),  sk.data(), sq.data(),  n,
                                                 1e-6f};
  miso::kernels::deltanet_chunk_kernel<256><<<32, 256>>>(chunk);
  MISO_HIP_CHECK(hipGetLastError());
  MISO_HIP_CHECK(hipDeviceSynchronize());

  const double ro = rel(o_ch.to_host(), o_seq.to_host());
  const double rs = rel(st_ch.to_host(), st_seq.to_host());
  const double rc = rel(co_ch.to_host(), co_seq.to_host());
  MESSAGE("n " << n << " out " << ro << " state " << rs << " conv " << rc);
  CHECK(rc == 0.0);
  CHECK(ro <= 1e-4);
  CHECK(rs <= 1e-4);
}

}  // namespace

TEST_CASE("chunked delta rule matches the decode recurrence, state included") {
  check(1);
  check(7);
  check(64);
  check(65);
  check(128);
}
