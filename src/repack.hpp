#pragma once

// Load-time weight repacking (ADR_001 D5): pure, deterministic, lossless functions of GGUF bytes.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "dequant.hpp"

namespace miso {

// Q4_K r1: same 144-byte block as GGUF Q4_K except bytes 4..15. There, nibble group g (sub-blocks
// 2g and 2g+1, whose nibbles share qs bytes 32g..32g+31) stores the 24-bit little-endian value
//   sc[2g] | m[2g] << 6 | sc[2g+1] << 12 | m[2g+1] << 18
// at bytes 4 + 3g, so a GPU lane extracts its four 6-bit fields without branches.
void repack_q4k(std::span<const std::byte> src, std::span<std::byte> dst);

// {sc[2g], m[2g], sc[2g+1], m[2g+1]} of an r1 block.
std::array<std::uint8_t, 4> q4k_r1_scales(const std::byte* block, int g);

inline constexpr std::size_t kQ6kBlockBytes = 210;

// Q6_K r1: each row of nb = row_elements / 256 GGUF Q6_K blocks (ql[128] qh[64] scales[16] d) is
// stored structure-of-arrays so every section is 16-byte aligned for vector loads:
//   [ql: nb x 128][qh: nb x 64][scales: nb x 16][d: nb x 2][zero padding to a multiple of 16]
std::size_t q6k_r1_row_bytes(std::size_t row_elements);
void repack_q6k(std::span<const std::byte> src, std::size_t row_elements, std::span<std::byte> dst);

// The original GGUF block b of an r1 row with nb blocks.
std::array<std::byte, kQ6kBlockBytes> q6k_r1_block(const std::byte* row, std::size_t nb,
                                                   std::size_t b);

}  // namespace miso
