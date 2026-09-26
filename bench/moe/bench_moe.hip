// Per-token time of one MoE block (router, 8 routed + shared experts) on the decode path. Inputs
// rotate over the 17 golden tokens so the selected experts change from call to call.
// Prefill: one 512-token chunk of N(0, 1) activations (near-uniform routing), median of 9 runs.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "engine/moe.hpp"

int main(int argc, char** argv) {
  const auto m = miso::gguf::File::open(argc > 1 ? argv[1] : MISO_ORNITH_GGUF);
  const auto g = miso::gguf::File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto* t = g.find_tensor("layer0.moe_in");
  const auto bytes = g.tensor_data(*t);
  miso::DeviceBuffer<float> x(bytes.size() / 4), y(2048);
  MISO_HIP_CHECK(hipMemcpy(x.data(), bytes.data(), bytes.size(), hipMemcpyHostToDevice));
  for (int layer : {0, 5}) {  // layer 0: Q6_K down; layer 5: Q4_K down
    const auto moe = miso::MoeLayer::load(m, layer);
    miso::MoeScratch sc;
    for (int i = 0; i < 200; ++i)
      miso::moe_decode(moe, sc, x.data() + (i % 17) * 2048, y.data(), nullptr);
    hipEvent_t a, b;
    MISO_HIP_CHECK(hipEventCreate(&a));
    MISO_HIP_CHECK(hipEventCreate(&b));
    const int n = 1000;
    MISO_HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < n; ++i)
      miso::moe_decode(moe, sc, x.data() + (i % 17) * 2048, y.data(), nullptr);
    MISO_HIP_CHECK(hipEventRecord(b));
    MISO_HIP_CHECK(hipEventSynchronize(b));
    float ms;
    MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    std::printf("MoE layer %d (down %s) decode: %.1f us/token\n", layer,
                moe.down_exps.type == miso::gguf::GgmlType::Q6_K ? "Q6_K" : "Q4_K", ms * 1e3 / n);

    constexpr unsigned kTok = 512;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hx(std::size_t{kTok} * 2048);
    for (auto& v : hx)
      v = nd(rng);
    miso::DeviceBuffer<float> px(hx), py(hx.size());
    miso::MoePrefillScratch ps(kTok);
    std::vector<float> runs(9);
    for (int w = 0; w < 3; ++w)
      miso::moe_prefill(moe, ps, px.data(), py.data(), kTok, nullptr);
    for (auto& r : runs) {
      MISO_HIP_CHECK(hipEventRecord(a));
      miso::moe_prefill(moe, ps, px.data(), py.data(), kTok, nullptr);
      MISO_HIP_CHECK(hipEventRecord(b));
      MISO_HIP_CHECK(hipEventSynchronize(b));
      MISO_HIP_CHECK(hipEventElapsedTime(&r, a, b));
    }
    std::sort(runs.begin(), runs.end());
    const double flops = 2.0 * kTok * 9 * 3 * 512 * 2048;
    std::printf("MoE layer %d prefill %u tokens: %.3f ms (%.1f us/token, %.2f TFLOP/s experts)\n",
                layer, kTok, runs[4], runs[4] * 1e3 / kTok, flops / (runs[4] * 1e9));
  }
  return 0;
}
