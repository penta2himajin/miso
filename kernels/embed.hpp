#pragma once

// Token embedding lookup from a Q4_K r1 table (src/repack.hpp): out[t] = dequant(table[ids[t]]).
// Bit-exact with the CPU reference: d * sc, d * sc * q and dmin * m are exact in FP32 for Q4_K, so
// the single rounding of the subtraction matches with or without FMA contraction.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace miso::kernels {

struct EmbedParams {
  const std::uint8_t* table;  // vocab rows of kN / 256 Q4_K r1 blocks
  const std::int32_t* ids;    // [tokens]
  float* out;                 // [tokens][kN]
};

// Tokens [first, first + count); one workgroup per token (grid-stride).
template <unsigned kN, int kBlock>
__device__ void embed_q4k_op(const EmbedParams& p, unsigned first, unsigned count) {
  static_assert(kN % 256 == 0);
  constexpr unsigned kRowBytes = kN / 256 * 144;
  for (unsigned t = first + blockIdx.x; t < first + count; t += gridDim.x) {
    const std::uint8_t* row = p.table + std::size_t(p.ids[t]) * kRowBytes;
    for (unsigned e = threadIdx.x; e < kN; e += kBlock) {
      const std::uint8_t* blk = row + (e / 256) * 144;
      const unsigned j = (e % 256) / 32, g = j / 2;
      unsigned v = 0;  // r1 24-bit field of nibble group g at bytes 4 + 3g
      for (int b = 0; b < 3; ++b)
        v |= unsigned{blk[4 + 3 * g + b]} << (8 * b);
      const unsigned sc = (j & 1 ? v >> 12 : v) & 63u, m = (j & 1 ? v >> 18 : v >> 6) & 63u;
      const unsigned q = (blk[16 + 32 * g + e % 32] >> (4 * (j & 1))) & 15u;
      const float d = static_cast<float>(*reinterpret_cast<const _Float16*>(blk));
      const float dmin = static_cast<float>(*reinterpret_cast<const _Float16*>(blk + 2));
      p.out[std::size_t{t} * kN + e] =
          __fsub_rn(__fmul_rn(__fmul_rn(d, static_cast<float>(sc)), static_cast<float>(q)),
                    __fmul_rn(dmin, static_cast<float>(m)));
    }
  }
}

template <unsigned kN, int kBlock>
__global__ void __launch_bounds__(kBlock) embed_q4k_kernel(EmbedParams p, unsigned tokens) {
  embed_q4k_op<kN, kBlock>(p, 0, tokens);
}

}  // namespace miso::kernels
