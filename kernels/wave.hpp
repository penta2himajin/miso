#pragma once

// Wavefront (64-lane) and workgroup reductions shared by kernels.

#include <hip/hip_runtime.h>

namespace miso::kernels {

constexpr int kWave = 64;

// DPP-moved copy of v; lanes the operation does not write (row mask / invalid source) get `old`.
template <int kCtrl, int kRowMask = 0xF>
__device__ inline float dpp(float v, float old = 0.0f) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, old), __builtin_bit_cast(int, v),
                                         kCtrl, kRowMask, 0xF, false));
}

// Reduction over the wavefront with `op`, `identity` filling unwritten lanes; valid in lane 63.
template <class Op>
__device__ inline float wave_reduce(float v, Op op, float identity) {
  v = op(v, dpp<0xB1>(v, identity));        // quad_perm [1,0,3,2]
  v = op(v, dpp<0x4E>(v, identity));        // quad_perm [2,3,0,1]: quads
  v = op(v, dpp<0x141>(v, identity));       // row_half_mirror: 8 lanes
  v = op(v, dpp<0x140>(v, identity));       // row_mirror: 16-lane rows
  v = op(v, dpp<0x142, 0xA>(v, identity));  // row_bcast:15 into rows 1, 3
  v = op(v, dpp<0x143, 0xC>(v, identity));  // row_bcast:31 into rows 2, 3
  return v;
}

__device__ inline float wave_sum(float v) {
  return wave_reduce(v, [](float a, float b) { return a + b; }, 0.0f);
}

__device__ inline float wave_max(float v) {
  return wave_reduce(v, [](float a, float b) { return fmaxf(a, b); }, -__builtin_inff());
}

// Reduction over a workgroup of kBlock threads, returned to every thread. `scratch` needs
// kBlock / 64 floats of LDS; the call contains two barriers.
template <int kBlock, class Op>
__device__ inline float block_reduce(float v, float* scratch, Op op, float identity) {
  static_assert(kBlock % kWave == 0);
  v = wave_reduce(v, op, identity);
  if (threadIdx.x % kWave == kWave - 1)
    scratch[threadIdx.x / kWave] = v;
  __syncthreads();
  float r = scratch[0];
#pragma unroll
  for (int w = 1; w < kBlock / kWave; ++w)
    r = op(r, scratch[w]);
  __syncthreads();
  return r;
}

template <int kBlock>
__device__ inline float block_sum(float v, float* scratch) {
  return block_reduce<kBlock>(v, scratch, [](float a, float b) { return a + b; }, 0.0f);
}

template <int kBlock>
__device__ inline float block_max(float v, float* scratch) {
  return block_reduce<kBlock>(
      v, scratch, [](float a, float b) { return fmaxf(a, b); }, -__builtin_inff());
}

}  // namespace miso::kernels
