#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/model.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

using miso::gguf::File;

namespace {

// Loading takes ~35 s; doctest re-runs the test body per SUBCASE, so load once.
const miso::Model& model() {
  static const File f = File::open(MISO_ORNITH_GGUF);
  static const miso::Model m = miso::Model::load(f);
  return m;
}

}  // namespace

// End to end over all 40 layers and the LM head, on llama.cpp's greedy continuation of the golden
// prompt (the reference also agrees 8/8, tests/golden/README.md):
//   teacher-forced: at every continuation position the engine's argmax is the next recorded token;
//   free-running:   greedy decoding after the prompt, feeding back the engine's own predictions on
//                   the device, generates exactly the recorded continuation.
TEST_CASE("full model decode agrees with llama.cpp's greedy continuation" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto& prompt = g.get<std::vector<std::int64_t>>("golden.token_ids");
  const auto& cont = g.get<std::vector<std::int64_t>>("golden.continuation_ids");
  REQUIRE(cont.size() == 8);
  std::vector<int> all(prompt.begin(), prompt.end());
  all.insert(all.end(), cont.begin(), cont.end());

  const miso::Model& m = model();
  CHECK(m.layers.size() == 40);

  SUBCASE("teacher-forced") {
    miso::Session s(m, 64);
    MISO_HIP_CHECK(hipMemcpy(s.tokens.data(), all.data(), all.size() * 4, hipMemcpyHostToDevice));
    for (unsigned t = 0; t + 1 < all.size(); ++t)
      miso::decode_step(m, s, s.tokens.data() + t, nullptr);
    const auto pred = s.pred.to_host();
    for (unsigned t = static_cast<unsigned>(prompt.size()) - 1; t + 1 < all.size(); ++t) {
      CAPTURE(t);
      CHECK(pred[t] == all[t + 1]);
    }
  }
  SUBCASE("free-running greedy") {
    miso::Session s(m, 64);
    MISO_HIP_CHECK(
        hipMemcpy(s.tokens.data(), all.data(), prompt.size() * 4, hipMemcpyHostToDevice));
    for (unsigned t = 0; t < prompt.size(); ++t)
      miso::decode_step(m, s, s.tokens.data() + t, nullptr);
    for (unsigned t = static_cast<unsigned>(prompt.size()); t < all.size() - 1; ++t) {
      miso::decode_step(m, s, s.pred.data() + t - 1, nullptr);
    }
    const auto pred = s.pred.to_host();
    for (unsigned k = 0; k < cont.size(); ++k) {
      CAPTURE(k);
      CHECK(pred[prompt.size() - 1 + k] == cont[k]);
    }
  }
}
