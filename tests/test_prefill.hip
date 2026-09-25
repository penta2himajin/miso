#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/model.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "session_compare.hpp"

using miso::gguf::File;

namespace {

const miso::Model& model() {
  static const File f = File::open(MISO_ORNITH_GGUF);
  static const miso::Model m = miso::Model::load(f);
  return m;
}

std::vector<int> prompt_tokens() {
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto& p = g.get<std::vector<std::int64_t>>("golden.token_ids");
  const auto& c = g.get<std::vector<std::int64_t>>("golden.continuation_ids");
  std::vector<int> t(p.begin(), p.end());
  t.insert(t.end(), c.begin(), c.end());
  return t;
}

}  // namespace

// Prefill (layer-major over chunks) must leave the same state and next-token logits as decoding
// the prompt token by token. With per-token kernels inside (M4a) the two are bit-identical; later
// milestones replace kernels and assert measured tolerances against this same oracle.
TEST_CASE("prefill matches token-by-token decode (oracle)" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const auto& m = model();
  const auto toks = prompt_tokens();
  const unsigned n = static_cast<unsigned>(toks.size());

  miso::Session want(m, 64);
  MISO_HIP_CHECK(hipMemcpy(want.tokens.data(), toks.data(), n * 4, hipMemcpyHostToDevice));
  for (unsigned t = 0; t < n; ++t)
    miso::decode_step(m, want, want.tokens.data() + t, nullptr);

  for (unsigned chunk : {n, 7u, 1u}) {
    CAPTURE(chunk);
    miso::Session got(m, 64);
    MISO_HIP_CHECK(hipMemcpy(got.tokens.data(), toks.data(), n * 4, hipMemcpyHostToDevice));
    miso::prefill(m, got, got.tokens.data(), n, chunk, nullptr);
    MISO_HIP_CHECK(hipDeviceSynchronize());
    const auto d = miso::test::compare(got, want);
    CHECK(d.same_pos);
    CHECK(d.same_pred);
    CHECK(d.max() == 0.0);

    // The state must be usable: one more decode step agrees too.
    miso::decode_step(m, got, got.pred.data() + got.pos - 1, nullptr);
    miso::Session want2(m, 64);  // fresh reference including the extra step
    MISO_HIP_CHECK(hipMemcpy(want2.tokens.data(), toks.data(), n * 4, hipMemcpyHostToDevice));
    for (unsigned t = 0; t < n; ++t)
      miso::decode_step(m, want2, want2.tokens.data() + t, nullptr);
    miso::decode_step(m, want2, want2.pred.data() + want2.pos - 1, nullptr);
    CHECK(miso::test::compare(got, want2).max() == 0.0);
  }
}
