// Per-token time of one Gated DeltaNet mixer on the decode path. Optional argv[2] selects the
// layer (default 0). Run under `rocprof --stats` for the per-kernel breakdown.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "engine/deltanet.hpp"

int main(int argc, char** argv) {
  const auto m = miso::gguf::File::open(argc > 1 ? argv[1] : MISO_ORNITH_GGUF);
  const int layer = argc > 2 ? std::atoi(argv[2]) : 0;
  const auto w = miso::DeltaNetLayer::load(m, layer);
  miso::DeltaNetState st;
  miso::DeltaNetScratch sc;
  miso::DeviceBuffer<float> h(std::vector<float>(2048, 0.1f)), y(2048);
  for (int i = 0; i < 200; ++i)
    miso::deltanet_decode(w, st, sc, h.data(), nullptr, y.data(), 1e-6f, nullptr);
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  const int n = 1000;
  MISO_HIP_CHECK(hipEventRecord(a));
  for (int i = 0; i < n; ++i)
    miso::deltanet_decode(w, st, sc, h.data(), nullptr, y.data(), 1e-6f, nullptr);
  MISO_HIP_CHECK(hipEventRecord(b));
  MISO_HIP_CHECK(hipEventSynchronize(b));
  float ms;
  MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  std::printf("DeltaNet layer %d decode: %.1f us/token (qkv.n=%u z_ab.n=%u, %d tokens)\n", layer,
              ms * 1e3 / n, w.qkv.n, w.z_ab.n, n);
  return 0;
}
