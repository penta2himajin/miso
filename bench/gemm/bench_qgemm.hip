// Prefill GEMM (kernels/qgemm.hpp) on real Ornith dense projections at T = 64 / 512 / 2048 tokens.
// Reports time per launch and TFLOP/s against the 25.4 TFLOP/s v_dot2_f32_f16 peak
// (docs/research/mi50.md), plus the FP16 activation prep as a separate launch.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "hip_util.hpp"
#include "qgemm.hpp"
#include "repack.hpp"

namespace {

using miso::kernels::QFormat;
constexpr double kPeakTflops = 25.4;

struct Weight {
  std::string name;
  QFormat format;
  unsigned n, k;
  miso::DeviceBuffer<std::uint8_t> data;
};

Weight upload(const miso::gguf::File& f, const std::string& name) {
  const auto* t = f.find_tensor(name);
  if (t == nullptr)
    throw std::runtime_error("missing " + name);
  const auto src = f.tensor_data(*t);
  const unsigned k = static_cast<unsigned>(t->dims[0]);
  const unsigned n = static_cast<unsigned>(t->n_elements() / k);
  std::vector<std::byte> r1;
  QFormat format;
  if (t->type == miso::gguf::GgmlType::Q4_K) {
    format = QFormat::Q4K;
    r1.resize(src.size());
    miso::repack_q4k(src, r1);
  } else {
    format = QFormat::Q6K;
    r1.resize(std::size_t{n} * miso::q6k_r1_row_bytes(k));
    miso::repack_q6k(src, k, r1);
  }
  std::vector<std::uint8_t> host(r1.size());
  std::memcpy(host.data(), r1.data(), r1.size());
  return {name, format, n, k, miso::DeviceBuffer<std::uint8_t>(host)};
}

template <class F>
float median_ms(F&& body, int trials = 7) {
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  for (int w = 0; w < 5; ++w)
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

}  // namespace

int main(int argc, char** argv) {
  const auto f = miso::gguf::File::open(argc > 1 ? argv[1] : MISO_ORNITH_GGUF);
  std::vector<Weight> ws;
  for (const char* name :
       {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
        "blk.3.attn_q.weight", "blk.0.ffn_up_shexp.weight"})
    ws.push_back(upload(f, name));

  constexpr unsigned kMaxT = 2048, kMaxK = 4096, kMaxN = 8192;
  miso::DeviceBuffer<float> x(std::vector<float>(std::size_t{kMaxT} * kMaxK, 0.01f));
  miso::DeviceBuffer<_Float16> xh(std::size_t{kMaxT} * kMaxK);
  miso::DeviceBuffer<float> xs(std::size_t{kMaxT} * kMaxK / 32), y(std::size_t{kMaxT} * kMaxN);

  std::printf("%-26s %4s %5s %5s %5s %9s %8s %6s\n", "weight", "fmt", "N", "K", "T", "ms",
              "TFLOP/s", "%peak");
  for (unsigned t : {64u, 512u, 2048u}) {
    for (unsigned k : {2048u, 4096u}) {
      const miso::kernels::QGemmActParams ap{x.data(), xh.data(), xs.data(), k, t};
      const float ms = median_ms([&] { miso::kernels::qgemm_act_kernel<<<240, 256>>>(ap); });
      std::printf("%-26s %4s %5s %5u %5u %9.4f  (%.0f GB/s)\n", "act prep", "-", "-", k, t, ms,
                  t * k * 10.0 / (ms * 1e6));
    }
    for (const auto& w : ws) {
      const miso::kernels::QGemmParams p{w.data.data(), xh.data(), xs.data(), y.data(),
                                         w.k,           w.n,       t,         w.n};
      const unsigned grid = miso::kernels::qgemm_tiles(w.n, t);
      const float ms = median_ms([&] {
        if (w.format == QFormat::Q4K)
          miso::kernels::qgemm_kernel<QFormat::Q4K><<<grid, 256>>>(p);
        else
          miso::kernels::qgemm_kernel<QFormat::Q6K><<<grid, 256>>>(p);
      });
      const double tflops = 2.0 * t * w.n * w.k / (ms * 1e9);
      std::printf("%-26s %4s %5u %5u %5u %9.4f %8.2f %5.1f%%\n", w.name.c_str(),
                  w.format == QFormat::Q4K ? "Q4_K" : "Q6_K", w.n, w.k, t, ms, tflops,
                  100 * tflops / kPeakTflops);
    }
  }
  MISO_HIP_CHECK(hipGetLastError());
  return 0;
}
