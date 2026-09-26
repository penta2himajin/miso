#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "engine/weights.hpp"
#include "hip_util.hpp"

namespace gguf = miso::gguf;

// The LoadStage path must produce byte-identical device weights to the plain pageable path
// (ADR_001 D5: both load paths produce byte-identical device buffers). This guards the M7f staging
// rewrite: repacking split across chunks and threads, and the copy out of pinned slots.
namespace {

// The smallest tensor of `type` that is still big enough to be split into several chunks.
const miso::gguf::TensorInfo* pick(const miso::gguf::File& f, miso::gguf::GgmlType type,
                                   std::size_t min_bytes) {
  const miso::gguf::TensorInfo* best = nullptr;
  for (const auto& t : f.tensors()) {
    if (t.type != type)
      continue;
    if (t.n_bytes() < min_bytes)
      continue;
    if (best == nullptr || t.n_bytes() < best->n_bytes())
      best = &t;
  }
  return best;
}

}  // namespace

TEST_CASE("staged uploads are byte-identical to the plain pageable path") {
  const auto f = miso::gguf::File::open(MISO_ORNITH_GGUF);
  miso::LoadStage stage;

  // Q4_K, Q6_K and F32 tensors, each loaded both ways.
  for (miso::gguf::GgmlType type :
       {miso::gguf::GgmlType::Q4_K, miso::gguf::GgmlType::Q6_K, miso::gguf::GgmlType::F32}) {
    const miso::gguf::TensorInfo* t = pick(f, type, 1u << 20);
    if (t == nullptr)
      continue;  // the model may not use every type
    CAPTURE(t->name);
    const auto plain = miso::upload_qmatrix(f, t->name, nullptr);
    const auto staged = miso::upload_qmatrix(f, t->name, &stage);
    stage.sync();
    CHECK(plain.n == staged.n);
    CHECK(plain.k == staged.k);
    CHECK(plain.type == staged.type);
    CHECK(plain.data.size() == staged.data.size());
    CHECK(plain.data.to_host() == staged.data.to_host());
  }
}

TEST_CASE("staged concatenation matches the plain path and preserves row order") {
  const auto f = miso::gguf::File::open(MISO_ORNITH_GGUF);
  miso::LoadStage stage;
  // attn_q and attn_k are both Q4_K in every attention layer, and their concatenation is the
  // row-order-sensitive case: the second tensor must land exactly after the first.
  const std::vector<std::string> names{"blk.3.attn_q.weight", "blk.3.attn_k.weight"};
  const auto plain = miso::concat_rows(f, names, nullptr);
  const auto staged = miso::concat_rows(f, names, &stage);
  stage.sync();
  CHECK(plain.n == staged.n);
  CHECK(staged.n == 8704);
  CHECK(plain.data.to_host() == staged.data.to_host());
}
