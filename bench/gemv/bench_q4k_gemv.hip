// Q4_K decode GEMV: bandwidth and accuracy of the INT8 and FP16 activation variants (ADR_002 D9).
//
// Each measurement launches the kernel back to back over the same projection of every layer, so
// every launch reads weights that are not in L2 (as in decode, where each layer is read once per
// token). Time per launch = elapsed / launches; bandwidth counts weight bytes only.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "dequant.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "q4k_gemv.hpp"
#include "repack.hpp"

using miso::kernels::ActFormat;

namespace {

constexpr int kBlock = 256;
constexpr double kHbmReadGBs = 797.0;  // docs/research/mi50.md §5.2

struct Projection {
  std::string name;
  unsigned n = 0, k = 0;
  std::vector<miso::DeviceBuffer<std::uint8_t>> layers;
  std::vector<float> ref_w;  // layer 0 dequantised, for accuracy
  std::size_t bytes = 0;     // per layer
};

std::vector<std::uint8_t> repacked(std::span<const std::byte> src) {
  std::vector<std::byte> r1(src.size());
  miso::repack_q4k(src, r1);
  const auto* b = reinterpret_cast<const std::uint8_t*>(r1.data());
  return {b, b + r1.size()};
}

Projection load(const miso::gguf::File& f, const std::string& suffix, int step, int first) {
  Projection p;
  p.name = suffix;
  for (int l = first; l < 40; l += step) {
    const auto* t = f.find_tensor("blk." + std::to_string(l) + "." + suffix);
    if (t == nullptr || t->type != miso::gguf::GgmlType::Q4_K)
      continue;
    const auto data = f.tensor_data(*t);
    p.k = static_cast<unsigned>(t->dims[0]);
    p.n = static_cast<unsigned>(t->n_elements() / p.k);
    p.bytes = data.size();
    p.layers.emplace_back(repacked(data));
    if (p.ref_w.empty()) {
      p.ref_w.resize(t->n_elements());
      miso::dequantize(t->type, data, p.ref_w);
    }
  }
  return p;
}

template <ActFormat A, unsigned K, int R>
void launch(const Projection& p, int layer, const float* x, float* y, unsigned grid) {
  miso::kernels::Q4kGemvParams prm{p.layers[layer].data(), x, y, p.n};
  miso::kernels::q4k_gemv_kernel<A, K, kBlock, R><<<grid, kBlock>>>(prm);
}

template <ActFormat A, unsigned K, int R>
void bench(const Projection& p, const std::vector<float>& xh, const char* label) {
  miso::DeviceBuffer<float> x(xh), y(p.n);
  // Accuracy on layer 0 against an FP64 reference: |err| / sum|w x| per row.
  launch<A, K, R>(p, 0, x.data(), y.data(), p.n / 4);
  const auto yh = y.to_host();
  double worst = 0, sq = 0;
  for (unsigned r = 0; r < p.n; ++r) {
    double ref = 0, mag = 0;
    for (unsigned k = 0; k < K; ++k) {
      const double wx = double{p.ref_w[std::size_t{r} * K + k]} * xh[k];
      ref += wx, mag += std::fabs(wx);
    }
    const double e = std::fabs(yh[r] - ref) / mag;
    worst = std::max(worst, e), sq += e * e;
  }
  std::printf("%-22s %-5s K=%-4u N=%-5u R=%d  rel.err max %.2e rms %.2e\n", p.name.c_str(), label,
              K, p.n, R, worst, std::sqrt(sq / p.n));

  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  const int n_layers = static_cast<int>(p.layers.size()), reps = 20;
  for (unsigned grid : {60u, 120u, 240u, 480u}) {
    // Warm-up long enough (~50 ms) for sclk to leave its idle level after host-side work.
    for (int r = 0; r < 100; ++r)
      for (int l = 0; l < n_layers; ++l)
        launch<A, K, R>(p, l, x.data(), y.data(), grid);
    std::vector<float> ms(5);
    for (auto& m : ms) {
      MISO_HIP_CHECK(hipEventRecord(a));
      for (int r = 0; r < reps; ++r)
        for (int l = 0; l < n_layers; ++l)
          launch<A, K, R>(p, l, x.data(), y.data(), grid);
      MISO_HIP_CHECK(hipEventRecord(b));
      MISO_HIP_CHECK(hipEventSynchronize(b));
      MISO_HIP_CHECK(hipEventElapsedTime(&m, a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double us = ms[2] * 1e3 / (reps * n_layers);
    const double gbs = p.bytes / (us * 1e3);
    std::printf("    grid %5u: %7.2f us/launch  %6.1f GB/s  %5.1f%% of %.0f\n", grid, us, gbs,
                100 * gbs / kHbmReadGBs, kHbmReadGBs);
  }
  MISO_HIP_CHECK(hipEventDestroy(a));
  MISO_HIP_CHECK(hipEventDestroy(b));
}

// All layers of a projection as one tall matrix in a single launch: the kernel's asymptotic
// bandwidth without per-launch ramp-up and tail.
template <ActFormat A, unsigned K, int R>
void bench_tall(const miso::gguf::File& f, const std::string& suffix, const std::vector<float>& xh,
                const char* label) {
  std::vector<std::uint8_t> all;
  unsigned n = 0;
  for (int l = 0; l < 40; ++l) {
    const auto* t = f.find_tensor("blk." + std::to_string(l) + "." + suffix);
    if (t == nullptr || t->type != miso::gguf::GgmlType::Q4_K)
      continue;
    const auto r1 = repacked(f.tensor_data(*t));
    all.insert(all.end(), r1.begin(), r1.end());
    n += static_cast<unsigned>(t->n_elements() / K);
  }
  miso::DeviceBuffer<std::uint8_t> w(all);
  miso::DeviceBuffer<float> x(xh), y(n);
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  for (unsigned grid : {240u, 480u, 960u, 1920u}) {
    miso::kernels::Q4kGemvParams prm{w.data(), x.data(), y.data(), n};
    miso::kernels::q4k_gemv_kernel<A, K, kBlock, R><<<grid, kBlock>>>(prm);
    std::vector<float> ms(5);
    for (auto& m : ms) {
      MISO_HIP_CHECK(hipEventRecord(a));
      miso::kernels::q4k_gemv_kernel<A, K, kBlock, R><<<grid, kBlock>>>(prm);
      MISO_HIP_CHECK(hipEventRecord(b));
      MISO_HIP_CHECK(hipEventSynchronize(b));
      MISO_HIP_CHECK(hipEventElapsedTime(&m, a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double gbs = all.size() / (ms[2] * 1e6);
    std::printf("tall %-17s %-5s R=%d N=%-6u %6.1f MB grid %5u: %8.1f us  %6.1f GB/s  %5.1f%%\n",
                suffix.c_str(), label, R, n, all.size() / 1e6, grid, ms[2] * 1e3, gbs,
                100 * gbs / kHbmReadGBs);
  }
  // Launch-size scaling: back-to-back launches of `rows` rows, each at a different offset in the
  // tall buffer (L2-cold). Fit t(rows) = t0 + bytes / BW to separate fixed cost from streaming.
  constexpr std::size_t kRowBytes = K / 256 * 144;
  for (unsigned rows : {480u, 960u, 1920u, 4096u, 8192u, 16384u, 32768u}) {
    const unsigned slots = n / rows, grid = 120;
    auto run = [&](int reps) {
      for (int r = 0; r < reps; ++r)
        for (unsigned s = 0; s < slots; ++s) {
          miso::kernels::Q4kGemvParams prm{w.data() + s * rows * kRowBytes, x.data(), y.data(),
                                           rows};
          miso::kernels::q4k_gemv_kernel<A, K, kBlock, R><<<grid, kBlock>>>(prm);
        }
    };
    run(20);
    float m;
    MISO_HIP_CHECK(hipEventRecord(a));
    run(10);
    MISO_HIP_CHECK(hipEventRecord(b));
    MISO_HIP_CHECK(hipEventSynchronize(b));
    MISO_HIP_CHECK(hipEventElapsedTime(&m, a, b));
    const double us = m * 1e3 / (10.0 * slots);
    std::printf("scale %-5s R=%d rows %6u (%6.2f MB) grid %u: %7.2f us/launch  %6.1f GB/s\n", label,
                R, rows, rows * kRowBytes / 1e6, grid, us, rows * kRowBytes / (us * 1e3));
  }
  MISO_HIP_CHECK(hipEventDestroy(a));
  MISO_HIP_CHECK(hipEventDestroy(b));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model = argc > 1 ? argv[1] : MISO_ORNITH_GGUF;
  const std::string golden = std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf";
  const auto f = miso::gguf::File::open(model);
  const auto g = miso::gguf::File::open(golden);
  const auto* moe_in = g.find_tensor("layer0.moe_in");
  const auto* xs = reinterpret_cast<const float*>(g.tensor_data(*moe_in).data());
  const std::vector<float> x2048(xs, xs + 2048), x4096(xs, xs + 4096);

  const auto gate = load(f, "attn_gate.weight", 1, 0);  // DeltaNet z: 2048 -> 4096, 30 layers
  const auto out = load(f, "ssm_out.weight", 1, 0);     // DeltaNet out: 4096 -> 2048, 30 layers
  const auto q = load(f, "attn_q.weight", 4, 3);        // full-attn q+gate: 2048 -> 8192, 10 layers
  std::printf("# %zu / %zu / %zu layers; %.2f / %.2f / %.2f MB per layer\n", gate.layers.size(),
              out.layers.size(), q.layers.size(), gate.bytes / 1e6, out.bytes / 1e6, q.bytes / 1e6);

  bench_tall<ActFormat::Int8, 2048, 1>(f, "attn_gate.weight", x2048, "int8");
  bench_tall<ActFormat::Int8, 2048, 4>(f, "attn_gate.weight", x2048, "int8");
  bench_tall<ActFormat::Fp16, 2048, 1>(f, "attn_gate.weight", x2048, "fp16");
  bench_tall<ActFormat::Fp16, 2048, 4>(f, "attn_gate.weight", x2048, "fp16");
  if (argc > 2 && std::string(argv[2]) == "--tall-only")
    return 0;

  bench<ActFormat::Int8, 2048, 1>(gate, x2048, "int8");
  bench<ActFormat::Int8, 2048, 4>(gate, x2048, "int8");
  bench<ActFormat::Fp16, 2048, 1>(gate, x2048, "fp16");
  bench<ActFormat::Fp16, 2048, 2>(gate, x2048, "fp16");
  bench<ActFormat::Fp16, 2048, 4>(gate, x2048, "fp16");
  bench<ActFormat::Fp16, 2048, 8>(gate, x2048, "fp16");
  bench<ActFormat::Int8, 4096, 4>(out, x4096, "int8");
  bench<ActFormat::Fp16, 4096, 1>(out, x4096, "fp16");
  bench<ActFormat::Fp16, 4096, 2>(out, x4096, "fp16");
  bench<ActFormat::Fp16, 4096, 4>(out, x4096, "fp16");
  bench<ActFormat::Int8, 2048, 4>(q, x2048, "int8");
  bench<ActFormat::Fp16, 2048, 4>(q, x2048, "fp16");
  return 0;
}
