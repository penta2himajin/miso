#pragma once

// Greedy sampling: index of the maximum logit (lowest index on ties, as torch.argmax), in two
// stages: per-workgroup partials over contiguous ranges, then one workgroup over the partials.

#include <hip/hip_runtime.h>

namespace miso::kernels {

namespace argmax_detail {

// (value, index) reduction within a workgroup through LDS; the result is in lane 0 of thread 0.
template <int kBlock>
__device__ inline void block_argmax(float& v, int& i) {
  __shared__ float sv[kBlock];
  __shared__ int si[kBlock];
  sv[threadIdx.x] = v, si[threadIdx.x] = i;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) {
      const float ov = sv[threadIdx.x + s];
      const int oi = si[threadIdx.x + s];
      if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
        sv[threadIdx.x] = ov, si[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  v = sv[0], i = si[0];
  __syncthreads();
}

}  // namespace argmax_detail

struct ArgmaxParams {
  const float* x;
  unsigned n;
  float* part_v;  // [gridDim.x of stage 1]
  int* part_i;
  unsigned n_parts;
  int* out;  // [1]
};

template <int kBlock>
__global__ void __launch_bounds__(kBlock) argmax_partial_kernel(ArgmaxParams p) {
  const unsigned per = (p.n + gridDim.x - 1) / gridDim.x;
  const unsigned lo = blockIdx.x * per, hi = min(p.n, lo + per);
  float v = -__builtin_inff();
  int idx = 0x7FFFFFFF;
  for (unsigned j = lo + threadIdx.x; j < hi; j += kBlock) {
    if (p.x[j] > v)
      v = p.x[j], idx = static_cast<int>(j);  // ascending j: keeps the lowest on ties
  }
  argmax_detail::block_argmax<kBlock>(v, idx);
  if (threadIdx.x == 0)
    p.part_v[blockIdx.x] = v, p.part_i[blockIdx.x] = idx;
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) argmax_final_kernel(ArgmaxParams p) {
  float v = -__builtin_inff();
  int idx = 0x7FFFFFFF;
  for (unsigned j = threadIdx.x; j < p.n_parts; j += kBlock) {
    if (p.part_v[j] > v || (p.part_v[j] == v && p.part_i[j] < idx))
      v = p.part_v[j], idx = p.part_i[j];
  }
  argmax_detail::block_argmax<kBlock>(v, idx);
  if (threadIdx.x == 0)
    *p.out = idx;
}

}  // namespace miso::kernels
