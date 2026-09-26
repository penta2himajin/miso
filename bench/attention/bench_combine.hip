// Isolate decode attention's partial-output combine, with the same split counts as production.
// Each variant sees identical inputs; alternating launch order controls warm-cache/clock bias.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "attention_decode.hpp"
#include "hip_util.hpp"

namespace k = miso::kernels;

int main() {
  std::mt19937 rng(71);
  std::uniform_real_distribution<float> values(-1.0f, 1.0f);
  std::vector<float> hm(2 * 129 * 8), hl(hm.size()), ho(hm.size() * 256), hg(16 * 512);
  for (auto& v : hm)
    v = values(rng) * 10;
  for (auto& v : hl)
    v = 1 + std::abs(values(rng)) * 16;
  for (auto& v : ho)
    v = values(rng);
  for (auto& v : hg)
    v = values(rng);
  miso::DeviceBuffer<float> m(hm), l(hl), o(ho), g(hg), core(16 * 256);
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  constexpr unsigned kRuns = 7, kWarm = 300, kSamples = 2000;
  for (unsigned ns : {1u, 16u, 22u, 57u, 60u, 65u, 129u}) {
    k::AttnCombineParams p{m.data(), l.data(), o.data(), g.data(), core.data(), ns};
    std::vector<float> times[4];
    for (unsigned run = 0; run < kRuns; ++run) {
      for (unsigned order = 0; order < 4; ++order) {
        const unsigned variant = (order + run) % 4;
        auto launch = [&] {
          if (variant == 0)
            k::attn_combine_kernel<256><<<16, 256>>>(p);
          else if (variant == 1)
            k::attn_combine_kernel<64><<<64, 64>>>(p);
          else if (variant == 2)
            k::attn_combine_kernel<64, true><<<64, 64>>>(p);
          else
            k::attn_combine_kernel<64, true, 4><<<64, 64>>>(p);
        };
        for (unsigned i = 0; i < kWarm; ++i)
          launch();
        MISO_HIP_CHECK(hipEventRecord(a));
        for (unsigned i = 0; i < kSamples; ++i)
          launch();
        MISO_HIP_CHECK(hipEventRecord(b));
        MISO_HIP_CHECK(hipEventSynchronize(b));
        float ms;
        MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
        times[variant].push_back(ms * 1000 / kSamples);
        std::printf("splits=%u run=%u variant=%u us=%.4f\n", ns, run + 1, variant,
                    times[variant].back());
      }
    }
    for (auto& t : times)
      std::sort(t.begin(), t.end());
    std::printf("median splits=%u block256=%.4f block64=%.4f prepared64=%.4f pipeline4=%.4f us\n",
                ns, times[0][kRuns / 2], times[1][kRuns / 2], times[2][kRuns / 2],
                times[3][kRuns / 2]);
  }
  MISO_HIP_CHECK(hipEventDestroy(a));
  MISO_HIP_CHECK(hipEventDestroy(b));
}
