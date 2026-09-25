#pragma once

// Helpers shared by the decode GEMV kernels.

#include <hip/hip_runtime.h>

namespace miso::kernels::gemv {

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

__device__ inline float fp16_bits(unsigned h) {
  return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(h)));
}

__device__ inline void load16(const float* p, float* v) {
  for (int c = 0; c < 4; ++c) {
    const float4 f = reinterpret_cast<const float4*>(p)[c];
    v[4 * c] = f.x, v[4 * c + 1] = f.y, v[4 * c + 2] = f.z, v[4 * c + 3] = f.w;
  }
}

__device__ inline half2_t to_half2(float a, float b) {
  return half2_t{static_cast<_Float16>(a), static_cast<_Float16>(b)};
}

// 16 activations as FP16 pairs in the order the nibble-to-half2 tricks produce weights:
// a[c] = (x[4c], x[4c+2]), b[c] = (x[4c+1], x[4c+3]); s16 is the sum of the rounded values.
struct Fp16x16 {
  half2_t a[4], b[4];
  float s16;
};

__device__ inline Fp16x16 fp16x16(const float* x) {
  float v[16];
  load16(x, v);
  Fp16x16 r;
  r.s16 = 0;
  for (int c = 0; c < 4; ++c) {
    r.a[c] = to_half2(v[4 * c], v[4 * c + 2]);
    r.b[c] = to_half2(v[4 * c + 1], v[4 * c + 3]);
    r.s16 += static_cast<float>(r.a[c].x) + static_cast<float>(r.a[c].y) +
             static_cast<float>(r.b[c].x) + static_cast<float>(r.b[c].y);
  }
  return r;
}

}  // namespace miso::kernels::gemv
