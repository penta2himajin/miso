// End-to-end decode throughput (ADR_002 D12): greedy generation with the full model, predictions
// fed back on the device so the host only enqueues. Reports model load time and tok/s after a
// prompt of the golden tokens, at the context lengths reached.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/model.hpp"

// Median of 3, after one discarded warm-up. Prints decode vs prefill milliseconds.
void sweep(const miso::Model& model, const std::vector<int>& prompt) {
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  auto once = [&](unsigned n, bool prefill_path) {
    std::vector<int> pp(n);
    for (unsigned i = 0; i < n; ++i)
      pp[i] = prompt[i % prompt.size()];
    miso::Session s(model, std::max(n, 8u) + 8);
    MISO_HIP_CHECK(hipMemcpy(s.tokens.data(), pp.data(), n * 4, hipMemcpyHostToDevice));
    MISO_HIP_CHECK(hipEventRecord(a));
    if (prefill_path)
      miso::prefill(model, s, s.tokens.data(), n, miso::Session::kMaxChunk, nullptr);
    else
      for (unsigned t = 0; t < n; ++t)
        miso::decode_step(model, s, s.tokens.data() + t, nullptr);
    MISO_HIP_CHECK(hipEventRecord(b));
    MISO_HIP_CHECK(hipEventSynchronize(b));
    float ms;
    MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms;
  };
  std::printf("n decode_ms prefill_ms\n");
  for (unsigned n : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u}) {
    auto med = [&](bool prefill_path) {
      once(n, prefill_path);
      float v[3];
      for (float& x : v)
        x = once(n, prefill_path);
      std::sort(v, v + 3);
      return v[1];
    };
    const float dec = med(false), pre = med(true);
    std::printf("%u %.2f %.2f\n", n, dec, pre);
  }
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "sweep") {
    const auto f = miso::gguf::File::open(MISO_ORNITH_GGUF);
    const auto model = miso::Model::load(f);
    MISO_HIP_CHECK(hipDeviceSynchronize());
    const auto g = miso::gguf::File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
    const auto& ids = g.get<std::vector<std::int64_t>>("golden.token_ids");
    sweep(model, std::vector<int>(ids.begin(), ids.end()));
    return 0;
  }

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
  {
    // pp512: a 512-token prompt (the golden tokens repeated) through prefill, as llama-bench does.
    std::vector<int> pp(512);
    for (std::size_t i = 0; i < pp.size(); ++i)
      pp[i] = prompt[i % prompt.size()];
    for (int round = 0; round < 2; ++round) {
      miso::Session sp(model, 520);
      MISO_HIP_CHECK(hipMemcpy(sp.tokens.data(), pp.data(), pp.size() * 4, hipMemcpyHostToDevice));
      MISO_HIP_CHECK(hipEventRecord(a));
      miso::prefill(model, sp, sp.tokens.data(), 512, miso::Session::kMaxChunk, nullptr);
      MISO_HIP_CHECK(hipEventRecord(b));
      MISO_HIP_CHECK(hipEventSynchronize(b));
      float ms;
      MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
      std::printf("%s: pp512 %.1f ms, %.1f tok/s\n", round ? "measured" : "warm-up", ms,
                  512e3 / ms);
    }
  }
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
