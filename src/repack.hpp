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

}  // namespace miso
