#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
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
// the prompt token by token, up to the rounding of two FP16-activation paths (ADR_004). Each layer
// type is held to 1e-3 of the golden reference on its own, and prefill and decode measure alike
// there: DeltaNet 2.96e-4, attention 4.63e-4, MoE 3.6e-4 / 3.9e-4 with bit-identical routing
// (test_deltanet, test_attention, test_moe).
//
// Over 40 layers the two paths can still route a token differently: a top-8 near-tie is decided
// by the rounding difference. On this prompt, layer 23 token 3 has its 8th and 9th router logits
// 5.8e-4 apart, 1e-4 of the logit range and below the FP16 activation error. From there the
// swapped expert propagates through that token's later layers and, via attention, to later
// tokens: 27 of 1000 token-layers route differently. Measured differences are logits 3.6e-3,
// KV 2.4e-2, conv 1.2e-2 and state 1.2e-2. Logic errors give 0.24 or more (wrong K offset, conv
// parity, a dropped MoE delta).
constexpr double kLogitsTolerance = 1e-2, kStateTolerance = 5e-2;

bool within(const miso::test::SessionDiff& d) {
  return d.logits <= kLogitsTolerance && std::max({d.kv, d.conv, d.state}) <= kStateTolerance;
}

TEST_CASE("prefill matches token-by-token decode (oracle)" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const auto& m = model();
  const auto toks = prompt_tokens();
  const unsigned n = static_cast<unsigned>(toks.size());

  miso::Session want(m, 64);
  MISO_HIP_CHECK(hipMemcpy(want.tokens.data(), toks.data(), n * 4, hipMemcpyHostToDevice));
  for (unsigned t = 0; t < n; ++t)
    miso::decode_step(m, want, want.tokens.data() + t, nullptr);

  // The reference including one more decode step after the prompt.
  miso::Session want2(m, 64);
  MISO_HIP_CHECK(hipMemcpy(want2.tokens.data(), toks.data(), n * 4, hipMemcpyHostToDevice));
  for (unsigned t = 0; t < n; ++t)
    miso::decode_step(m, want2, want2.tokens.data() + t, nullptr);
  miso::decode_step(m, want2, want2.pred.data() + want2.pos - 1, nullptr);

  std::optional<miso::Session> whole;  // the prompt as one chunk
  for (unsigned chunk : {n, 7u, 1u}) {
    CAPTURE(chunk);
    miso::Session got(m, 64);
    MISO_HIP_CHECK(hipMemcpy(got.tokens.data(), toks.data(), n * 4, hipMemcpyHostToDevice));
    miso::prefill(m, got, got.tokens.data(), n, chunk, nullptr);
    MISO_HIP_CHECK(hipDeviceSynchronize());
    const auto d = miso::test::compare(got, want);
    MESSAGE("logits " << d.logits << " kv " << d.kv << " conv " << d.conv << " state " << d.state);
    CHECK(d.same_pos);
    CHECK(d.same_pred);
    CHECK(within(d));
    // Every token's projections are computed independently of the chunk it lands in.
    if (whole)
      CHECK(miso::test::compare(got, *whole).max() == 0.0);

    // The state must be usable: one more decode step agrees too.
    miso::decode_step(m, got, got.pred.data() + got.pos - 1, nullptr);
    const auto d2 = miso::test::compare(got, want2);
    CHECK(d2.same_pred);
    CHECK(within(d2));
    if (!whole) {
      whole.emplace(m, 64);
      MISO_HIP_CHECK(hipMemcpy(whole->tokens.data(), toks.data(), n * 4, hipMemcpyHostToDevice));
      miso::prefill(m, *whole, whole->tokens.data(), n, n, nullptr);
    }
  }
}
