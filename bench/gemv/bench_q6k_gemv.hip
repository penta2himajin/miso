// Q6_K decode GEMV (FP16 activations): per-launch time over the layer-rotated DeltaNet qkv
// projections (L2-cold, as in decode) and the LM head as one 417 MB launch.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "hip_util.hpp"
#include "q6k_gemv.hpp"
#include "repack.hpp"

namespace {

constexpr int kBlock = 256;
constexpr double kHbmReadGBs = 797.0;

miso::DeviceBuffer<std::uint8_t> upload_r1(const miso::gguf::File& f,
                                           const miso::gguf::TensorInfo& t) {
  const auto src = f.tensor_data(t);
  const std::size_t k = t.dims[0], rows = t.n_elements() / k;
  std::vector<std::byte> r1(rows * miso::q6k_r1_row_bytes(k));
  miso::repack_q6k(src, k, r1);
  const auto* b = reinterpret_cast<const std::uint8_t*>(r1.data());
  return miso::DeviceBuffer<std::uint8_t>(std::vector<std::uint8_t>(b, b + r1.size()));
}

template <class F>
float median_ms(F&& body, int trials = 5) {
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  for (int w = 0; w < 10; ++w)
    body();  // warm-up so sclk leaves idle
  std::vector<float> ms(trials);
  for (auto& m : ms) {
    MISO_HIP_CHECK(hipEventRecord(a));
    body();
    MISO_HIP_CHECK(hipEventRecord(b));
    MISO_HIP_CHECK(hipEventSynchronize(b));
    MISO_HIP_CHECK(hipEventElapsedTime(&m, a, b));
  }
  std::sort(ms.begin(), ms.end());
  MISO_HIP_CHECK(hipEventDestroy(a));
  MISO_HIP_CHECK(hipEventDestroy(b));
  return ms[trials / 2];
}

template <int R>
void bench_layers(const std::vector<miso::DeviceBuffer<std::uint8_t>>& layers, unsigned n,
                  std::size_t bytes, const float* x, float* y) {
  for (unsigned grid : {120u, 240u, 480u}) {
    const int reps = 20;
    const float ms = median_ms([&] {
      for (int r = 0; r < reps; ++r)
        for (const auto& l : layers) {
          miso::kernels::Q6kGemvParams p{l.data(), x, y, n};
          miso::kernels::q6k_gemv_kernel<2048, kBlock, R><<<grid, kBlock>>>(p);
        }
    });
    const double us = ms * 1e3 / (reps * layers.size());
    std::printf("attn_qkv Q6_K  R=%d grid %4u: %7.2f us/launch  %6.1f GB/s  %5.1f%%\n", R, grid, us,
                bytes / (us * 1e3), 100 * bytes / (us * 1e3) / kHbmReadGBs);
  }
}

template <int R>
void bench_head(const miso::DeviceBuffer<std::uint8_t>& head, unsigned n, std::size_t bytes,
                const float* x, float* y) {
  for (unsigned grid : {240u, 480u, 960u, 1920u}) {
    const float ms = median_ms([&] {
      miso::kernels::Q6kGemvParams p{head.data(), x, y, n};
      miso::kernels::q6k_gemv_kernel<2048, kBlock, R><<<grid, kBlock>>>(p);
    });
    std::printf("LM head Q6_K   R=%d grid %4u: %8.1f us  %6.1f GB/s  %5.1f%%\n", R, grid, ms * 1e3,
                bytes / (ms * 1e6), 100 * bytes / (ms * 1e6) / kHbmReadGBs);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const auto f = miso::gguf::File::open(argc > 1 ? argv[1] : MISO_ORNITH_GGUF);
  std::vector<miso::DeviceBuffer<std::uint8_t>> qkv;
  std::size_t qkv_bytes = 0;
  unsigned qkv_n = 0;
  for (const auto& t : f.tensors()) {
    if (t.type != miso::gguf::GgmlType::Q6_K || t.name.find(".attn_qkv.") == std::string::npos)
      continue;
    qkv.push_back(upload_r1(f, t));
    qkv_bytes = qkv.back().size();
    qkv_n = static_cast<unsigned>(t.n_elements() / 2048);
  }
  const auto* ht = f.find_tensor("output.weight");
  const auto head = upload_r1(f, *ht);
  const unsigned head_n = static_cast<unsigned>(ht->n_elements() / 2048);
  std::printf("# %zu Q6_K attn_qkv layers x %.2f MB; LM head %.1f MB\n", qkv.size(),
              qkv_bytes / 1e6, head.size() / 1e6);

  miso::DeviceBuffer<float> x(std::vector<float>(2048, 0.01f)), y(head_n);
  bench_layers<1>(qkv, qkv_n, qkv_bytes, x.data(), y.data());
  bench_layers<2>(qkv, qkv_n, qkv_bytes, x.data(), y.data());
  bench_layers<4>(qkv, qkv_n, qkv_bytes, x.data(), y.data());
  bench_head<1>(head, head_n, head.size(), x.data(), y.data());
  bench_head<4>(head, head_n, head.size(), x.data(), y.data());
  return 0;
}
