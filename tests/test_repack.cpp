#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "dequant.hpp"
#include "gguf.hpp"
#include "repack.hpp"

using miso::gguf::File;

namespace {

// Checks that repacking every block of `src` is lossless and only moves the scale bytes.
void check_blocks(std::span<const std::byte> src) {
  REQUIRE(src.size() % miso::kQ4kBlockBytes == 0);
  std::vector<std::byte> dst(src.size());
  miso::repack_q4k(src, dst);
  std::size_t bad = 0;
  for (std::size_t off = 0; off < src.size(); off += miso::kQ4kBlockBytes) {
    const std::byte* s = src.data() + off;
    const std::byte* d = dst.data() + off;
    bad += std::memcmp(s, d, 4) != 0;              // d, dmin
    bad += std::memcmp(s + 16, d + 16, 128) != 0;  // nibbles
    for (int g = 0; g < 4; ++g) {
      std::uint8_t sc0, m0, sc1, m1;
      miso::q4k_scale_min(s, 2 * g, sc0, m0);
      miso::q4k_scale_min(s, 2 * g + 1, sc1, m1);
      const auto r = miso::q4k_r1_scales(d, g);
      bad += r[0] != sc0 || r[1] != m0 || r[2] != sc1 || r[3] != m1;
    }
  }
  CHECK(bad == 0);
}

}  // namespace

TEST_CASE("Q4_K r1 repack is lossless (fixture blocks)") {
  const File f = File::open(std::string(MISO_FIXTURE_DIR) + "/tiny.gguf");
  const auto* t = f.find_tensor("q4_k");
  REQUIRE(t != nullptr);
  check_blocks(f.tensor_data(*t));
}

TEST_CASE("Q4_K r1 repack is lossless (every Q4_K tensor of the model)" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  const File m = File::open(MISO_ORNITH_GGUF);
  std::size_t tensors = 0;
  for (const auto& t : m.tensors()) {
    if (t.type != miso::gguf::GgmlType::Q4_K || t.name.find("_exps") != std::string::npos)
      continue;
    CAPTURE(t.name);
    check_blocks(m.tensor_data(t));
    ++tensors;
  }
  CHECK(tensors > 200);
}

TEST_CASE("repack_q4k rejects mismatched sizes") {
  std::vector<std::byte> a(miso::kQ4kBlockBytes), b(miso::kQ4kBlockBytes + 1);
  CHECK_THROWS(miso::repack_q4k(a, b));
}
