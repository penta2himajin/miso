// Per-token time of one gated full-attention mixer (layer 3) on the decode path, as the KV cache
// grows. Run under `rocprof --stats` for the per-kernel breakdown.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#include "engine/attention.hpp"

int main(int argc, char** argv) {
  const auto m = miso::gguf::File::open(argc > 1 ? argv[1] : MISO_ORNITH_GGUF);
  const auto layer = miso::AttentionLayer::load(m, 3);
  const auto rope = miso::RopeConfig::from(m);
  const unsigned max_ctx = 16384 + 256;
  miso::AttentionCache cache(max_ctx);
  miso::AttentionScratch sc;
  miso::DeviceBuffer<float> h(std::vector<float>(2048, 0.1f)), y(2048);
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  for (unsigned ctx : {16u, 1024u, 4096u, 16384u}) {
    while (cache.len < ctx)
      miso::attention_decode(layer, cache, sc, rope, h.data(), y.data(), 1e-6f, nullptr);
    const int n = 64;
    MISO_HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < n; ++i)
      miso::attention_decode(layer, cache, sc, rope, h.data(), y.data(), 1e-6f, nullptr);
    MISO_HIP_CHECK(hipEventRecord(b));
    MISO_HIP_CHECK(hipEventSynchronize(b));
    float ms;
    MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    std::printf("attention layer 3 decode at ctx %5u: %8.1f us/token\n", ctx, ms * 1e3 / n);
  }
  return 0;
}
