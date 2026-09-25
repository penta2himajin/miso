#include "repack.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace miso {

void repack_q4k(std::span<const std::byte> src, std::span<std::byte> dst) {
  if (src.size() != dst.size() || src.size() % kQ4kBlockBytes != 0) {
    throw std::runtime_error("repack_q4k: sizes " + std::to_string(src.size()) + " -> " +
                             std::to_string(dst.size()) + " are not equal whole blocks");
  }
  for (std::size_t off = 0; off < src.size(); off += kQ4kBlockBytes) {
    const std::byte* s = src.data() + off;
    std::byte* d = dst.data() + off;
    std::memcpy(d, s, kQ4kBlockBytes);
    for (int g = 0; g < 4; ++g) {
      std::uint8_t sc0, m0, sc1, m1;
      q4k_scale_min(s, 2 * g, sc0, m0);
      q4k_scale_min(s, 2 * g + 1, sc1, m1);
      const std::uint32_t v = sc0 | m0 << 6 | sc1 << 12 | static_cast<std::uint32_t>(m1) << 18;
      for (int b = 0; b < 3; ++b)
        d[4 + 3 * g + b] = static_cast<std::byte>(v >> (8 * b));
    }
  }
}

std::array<std::uint8_t, 4> q4k_r1_scales(const std::byte* block, int g) {
  std::uint32_t v = 0;
  for (int b = 0; b < 3; ++b)
    v |= std::to_integer<std::uint32_t>(block[4 + 3 * g + b]) << (8 * b);
  return {static_cast<std::uint8_t>(v & 63), static_cast<std::uint8_t>((v >> 6) & 63),
          static_cast<std::uint8_t>((v >> 12) & 63), static_cast<std::uint8_t>((v >> 18) & 63)};
}

}  // namespace miso
