#include <doctest/doctest.h>

#include <filesystem>

#include "engine/model.hpp"
#include "gguf.hpp"

// The locally built pure-Q4_K file (docs/research/ornith-pure-q4k.md). Every quantised tensor is
// Q4_K; the engine must load it, including tensors that Q4_K_M keeps at Q6_K.
TEST_CASE("pure Q4_K GGUF loads" * doctest::skip(!std::filesystem::exists(MISO_ORNITH_Q4K))) {
  const auto f = miso::gguf::File::open(MISO_ORNITH_Q4K);
  unsigned q4 = 0, other = 0;
  for (const auto& t : f.tensors()) {
    using miso::gguf::GgmlType;
    if (t.type == GgmlType::F32 || t.type == GgmlType::F16 || t.type == GgmlType::BF16 ||
        t.type == GgmlType::I8 || t.type == GgmlType::I16 || t.type == GgmlType::I32 ||
        t.type == GgmlType::I64 || t.type == GgmlType::F64)
      continue;
    if (t.type == GgmlType::Q4_K) {
      ++q4;
    } else {
      ++other;
      MESSAGE(t.name << " type " << static_cast<unsigned>(t.type));
    }
  }
  CHECK(q4 > 0);
  CHECK(other == 0);

  const auto m = miso::Model::load(f);
  CHECK(m.lm_head.type == miso::gguf::GgmlType::Q4_K);
  CHECK(m.layers[0].moe.down_exps.type == miso::gguf::GgmlType::Q4_K);
  CHECK(m.layers[0].deltanet->qkv.type == miso::gguf::GgmlType::Q4_K);
}
