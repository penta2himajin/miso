// Per-token time of one Gated DeltaNet mixer (layer 0) on the decode path. Run under
// `rocprof --stats` for the per-kernel breakdown.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#include "engine/deltanet.hpp"

int main(int argc, char** argv) {
  const auto m = miso::gguf::File::open(argc > 1 ? argv[1] : MISO_ORNITH_GGUF);
  const auto layer = miso::DeltaNetLayer::load(m, 0);
  miso::DeltaNetState st;
  miso::DeltaNetScratch sc;
  miso::DeviceBuffer<float> h(std::vector<float>(2048, 0.1f)), y(2048);
  for (int i = 0; i < 200; ++i)
    miso::deltanet_decode(layer, st, sc, h.data(), y.data(), 1e-6f, nullptr);
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  const int n = 1000;
  MISO_HIP_CHECK(hipEventRecord(a));
  for (int i = 0; i < n; ++i)
    miso::deltanet_decode(layer, st, sc, h.data(), y.data(), 1e-6f, nullptr);
  MISO_HIP_CHECK(hipEventRecord(b));
  MISO_HIP_CHECK(hipEventSynchronize(b));
  float ms;
  MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  std::printf("DeltaNet layer 0 decode: %.1f us/token (%d tokens back to back)\n", ms * 1e3 / n, n);
  return 0;
}
