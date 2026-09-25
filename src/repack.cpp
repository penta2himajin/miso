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

namespace {

// Byte offset of each section of a Q6_K r1 row with nb blocks.
struct Q6kSections {
  std::size_t ql, qh, sc, d;
};
Q6kSections q6k_sections(std::size_t nb) {
  return {0, nb * 128, nb * 192, nb * 208};
}

}  // namespace

std::size_t q6k_r1_row_bytes(std::size_t row_elements) {
  return (row_elements / 256 * kQ6kBlockBytes + 15) / 16 * 16;
}

void repack_q6k(std::span<const std::byte> src, std::size_t row_elements,
                std::span<std::byte> dst) {
  if (row_elements == 0 || row_elements % 256 != 0) {
    throw std::runtime_error("repack_q6k: row length " + std::to_string(row_elements) +
                             " is not a multiple of 256");
  }
  const std::size_t nb = row_elements / 256, src_row = nb * kQ6kBlockBytes;
  const std::size_t dst_row = q6k_r1_row_bytes(row_elements);
  if (src.size() % src_row != 0 || dst.size() != src.size() / src_row * dst_row) {
    throw std::runtime_error("repack_q6k: " + std::to_string(src.size()) + " -> " +
                             std::to_string(dst.size()) + " bytes do not match whole rows");
  }
  const Q6kSections s = q6k_sections(nb);
  for (std::size_t r = 0; r < src.size() / src_row; ++r) {
    std::byte* out = dst.data() + r * dst_row;
    std::memset(out, 0, dst_row);
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte* blk = src.data() + r * src_row + b * kQ6kBlockBytes;
      std::memcpy(out + s.ql + b * 128, blk, 128);
      std::memcpy(out + s.qh + b * 64, blk + 128, 64);
      std::memcpy(out + s.sc + b * 16, blk + 192, 16);
      std::memcpy(out + s.d + b * 2, blk + 208, 2);
    }
  }
}

std::array<std::byte, kQ6kBlockBytes> q6k_r1_block(const std::byte* row, std::size_t nb,
                                                   std::size_t b) {
  const Q6kSections s = q6k_sections(nb);
  std::array<std::byte, kQ6kBlockBytes> blk;
  std::memcpy(blk.data(), row + s.ql + b * 128, 128);
  std::memcpy(blk.data() + 128, row + s.qh + b * 64, 64);
  std::memcpy(blk.data() + 192, row + s.sc + b * 16, 16);
  std::memcpy(blk.data() + 208, row + s.d + b * 2, 2);
  return blk;
}

}  // namespace miso
