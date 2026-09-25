#pragma once

// CPU reference dequantisation. Bit-exact with gguf-py (`gguf.quants.dequantize`), which is the
// source of the golden outputs (ADR_002 D11).

#include <cstddef>
#include <cstdint>
#include <span>

#include "gguf.hpp"

namespace miso {

// Supported: F32, F16, BF16, Q4_K, Q6_K. `src` must hold exactly dst.size() elements of `type`.
void dequantize(gguf::GgmlType type, std::span<const std::byte> src, std::span<float> dst);

inline constexpr std::size_t kQ4kBlockBytes = 144;

// IEEE binary16 bits to FP32 (exact, including subnormals, infinities and NaN).
float fp16_to_f32(std::uint16_t h);

// 6-bit scale and min of sub-block j (0..7) of a GGUF Q4_K block.
void q4k_scale_min(const std::byte* block, int j, std::uint8_t& sc, std::uint8_t& m);

}  // namespace miso
