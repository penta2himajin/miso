// End-to-end decode throughput (ADR_002 D12): greedy generation with the full model, predictions
// fed back on the device so the host only enqueues. Reports model load time and tok/s after a
// prompt of the golden tokens, at the context lengths reached.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/model.hpp"

int main(int argc, char** argv) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto f = miso::gguf::File::open(argc > 1 ? argv[1] : MISO_ORNITH_GGUF);
  const auto model = miso::Model::load(f);
  MISO_HIP_CHECK(hipDeviceSynchronize());
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("model load: %.1f s\n", load_s);

  const auto g = miso::gguf::File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto& ids = g.get<std::vector<std::int64_t>>("golden.token_ids");
  const std::vector<int> prompt(ids.begin(), ids.end());
  const unsigned gen = 256;
  miso::Session s(model, static_cast<unsigned>(prompt.size()) + 2 * gen + 8);
  MISO_HIP_CHECK(
      hipMemcpy(s.tokens.data(), prompt.data(), prompt.size() * 4, hipMemcpyHostToDevice));
  for (unsigned t = 0; t < prompt.size(); ++t)
    miso::decode_step(model, s, s.tokens.data() + t, nullptr);

  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  for (int round = 0; round < 2; ++round) {  // round 0 warms up clocks
    const unsigned start = s.pos;
    MISO_HIP_CHECK(hipEventRecord(a));
    for (unsigned k = 0; k < gen; ++k)
      miso::decode_step(model, s, s.pred.data() + s.pos - 1, nullptr);
    MISO_HIP_CHECK(hipEventRecord(b));
    MISO_HIP_CHECK(hipEventSynchronize(b));
    float ms;
    MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    std::printf("%s: %u tokens at ctx %u..%u: %.2f ms/token, %.1f tok/s\n",
                round ? "measured" : "warm-up", gen, start, s.pos, ms / gen, gen * 1e3 / ms);
  }
  const auto pred = s.pred.to_host();
  std::printf("first generated ids:");
  for (unsigned k = 0; k < 8; ++k)
    std::printf(" %d", pred[prompt.size() - 1 + k]);
  std::printf("\n");
  return 0;
}
