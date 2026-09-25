#pragma once

// Wavefront (64-lane) and workgroup reductions shared by kernels.

#include <hip/hip_runtime.h>

namespace miso::kernels {

constexpr int kWave = 64;

template <int kCtrl, int kRowMask = 0xF>
__device__ inline float dpp(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), kCtrl,
                                                               kRowMask, 0xF, false));
}

// Sum over the wavefront; the result is valid in lane 63.
__device__ inline float wave_sum(float v) {
  v += dpp<0xB1>(v);        // quad_perm [1,0,3,2]
  v += dpp<0x4E>(v);        // quad_perm [2,3,0,1]: quad sums
  v += dpp<0x141>(v);       // row_half_mirror: 8-lane sums
  v += dpp<0x140>(v);       // row_mirror: 16-lane row sums
  v += dpp<0x142, 0xA>(v);  // row_bcast:15 into rows 1, 3
  v += dpp<0x143, 0xC>(v);  // row_bcast:31 into rows 2, 3
  return v;
}

// Sum over a workgroup of kBlock threads, returned to every thread. `scratch` needs kBlock / 64
// floats of LDS; the call contains two barriers.
template <int kBlock>
__device__ inline float block_sum(float v, float* scratch) {
  static_assert(kBlock % kWave == 0);
  v = wave_sum(v);
  if (threadIdx.x % kWave == kWave - 1)
    scratch[threadIdx.x / kWave] = v;
  __syncthreads();
  float s = 0;
#pragma unroll
  for (int w = 0; w < kBlock / kWave; ++w)
    s += scratch[w];
  __syncthreads();
  return s;
}

}  // namespace miso::kernels
