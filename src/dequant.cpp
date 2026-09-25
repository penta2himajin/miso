#include "dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace miso {

namespace {

using gguf::GgmlType;

std::uint16_t load_u16(const std::byte* p) {
  std::uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

float bits_to_f32(std::uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

}  // namespace

float fp16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x3FFu;
  if (exp == 0) {
    const float v = static_cast<float>(mant) * 0x1p-24f;  // exact: zero or subnormal
    return sign ? -v : v;
  }
  if (exp == 31)
    return bits_to_f32(sign | 0x7F800000u | (mant << 13));
  return bits_to_f32(sign | ((exp + 112) << 23) | (mant << 13));
}

namespace {

constexpr int kQK = 256;

// Q4_K block: f16 d, f16 dmin, 12 bytes of packed 6-bit scales/mins, 128 bytes of nibbles.
void dequant_q4_k(const std::byte* b, float* y) {
  const float d = fp16_to_f32(load_u16(b));
  const float dmin = fp16_to_f32(load_u16(b + 2));
  const auto* qs = reinterpret_cast<const std::uint8_t*>(b + 16);
  for (int j = 0; j < 8; ++j) {
    std::uint8_t sc, m;
    q4k_scale_min(b, j, sc, m);
    const float dj = d * static_cast<float>(sc);
    const float mj = dmin * static_cast<float>(m);
    const int g = j / 2, shift = 4 * (j % 2);
    for (int l = 0; l < 32; ++l) {
      const int q = (qs[g * 32 + l] >> shift) & 0x0F;
      y[j * 32 + l] = dj * static_cast<float>(q) - mj;
    }
  }
}

// Q6_K block: 128 bytes low nibbles, 64 bytes high 2-bit pairs, 16 int8 scales, f16 d.
void dequant_q6_k(const std::byte* b, float* y) {
  const auto* ql = reinterpret_cast<const std::uint8_t*>(b);
  const auto* qh = reinterpret_cast<const std::uint8_t*>(b + 128);
  const auto* sc = reinterpret_cast<const std::int8_t*>(b + 192);
  const float d = fp16_to_f32(load_u16(b + 208));
  for (int e = 0; e < kQK; ++e) {
    const int h = e / 128, r = e % 128;
    const int lo = (ql[h * 64 + r % 64] >> (4 * (r / 64))) & 0x0F;
    const int hi = (qh[h * 32 + r % 32] >> (2 * (r / 32))) & 0x03;
    const float ds = d * static_cast<float>(sc[e / 16]);
    y[e] = ds * static_cast<float>((lo | (hi << 4)) - 32);
  }
}

}  // namespace

void q4k_scale_min(const std::byte* block, int j, std::uint8_t& sc, std::uint8_t& m) {
  const auto* s = reinterpret_cast<const std::uint8_t*>(block + 4);
  sc = j < 4 ? s[j] & 0x3F : (s[j + 4] & 0x0F) | ((s[j - 4] >> 2) & 0x30);
  m = j < 4 ? s[j + 4] & 0x3F : (s[j + 4] >> 4) | ((s[j] >> 2) & 0x30);
}

void dequantize(GgmlType type, std::span<const std::byte> src, std::span<float> dst) {
  const auto& tr = gguf::traits(type);
  const std::size_t n = dst.size();
  if (n % tr.block_elements != 0 || src.size() != n / tr.block_elements * tr.block_bytes) {
    throw std::runtime_error(std::string("dequantize ") + tr.name + ": " +
                             std::to_string(src.size()) + " bytes do not hold " +
                             std::to_string(n) + " elements");
  }
  const std::byte* p = src.data();
  float* y = dst.data();
  switch (type) {
    case GgmlType::F32:
      std::memcpy(y, p, n * 4);
      return;
    case GgmlType::F16:
      for (std::size_t i = 0; i < n; ++i)
        y[i] = fp16_to_f32(load_u16(p + 2 * i));
      return;
    case GgmlType::BF16:
      for (std::size_t i = 0; i < n; ++i)
        y[i] = bits_to_f32(std::uint32_t{load_u16(p + 2 * i)} << 16);
      return;
    case GgmlType::Q4_K:
      for (std::size_t i = 0; i < n / kQK; ++i)
        dequant_q4_k(p + i * 144, y + i * kQK);
      return;
    case GgmlType::Q6_K:
      for (std::size_t i = 0; i < n / kQK; ++i)
        dequant_q6_k(p + i * 210, y + i * kQK);
      return;
    default:
      throw std::runtime_error(std::string("dequantize: unsupported type ") + tr.name);
  }
}

}  // namespace miso
