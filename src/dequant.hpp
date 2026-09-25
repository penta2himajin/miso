#pragma once

// CPU reference dequantisation. Bit-exact with gguf-py (`gguf.quants.dequantize`), which is the
// source of the golden outputs (ADR_002 D11).

#include <span>

#include "gguf.hpp"

namespace miso {

// Supported: F32, F16, BF16, Q4_K, Q6_K. `src` must hold exactly dst.size() elements of `type`.
void dequantize(gguf::GgmlType type, std::span<const std::byte> src, std::span<float> dst);

}  // namespace miso
