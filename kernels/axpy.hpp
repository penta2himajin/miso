#pragma once

// Smoke-test operator (y = a * x + y) that fixes the operator pattern of ADR_001 "D4 ->
// megakernel": the body is a __device__ function over an explicit work-item range, and the
// __global__ kernel is a thin grid-stride wrapper, so the body can later run inside a persistent
// kernel unchanged.

#include <hip/hip_runtime.h>

namespace miso::kernels {

struct AxpyParams {
  float a;
  const float* x;
  float* y;
  unsigned n;
};

// Processes work items [first, first + count); one work item is one element.
template <int kBlock>
__device__ void axpy_op(const AxpyParams& p, unsigned first, unsigned count) {
  const unsigned end = first + count;
  for (unsigned i = first + blockIdx.x * kBlock + threadIdx.x; i < end; i += gridDim.x * kBlock) {
    p.y[i] = p.a * p.x[i] + p.y[i];
  }
}

template <int kBlock>
__global__ void __launch_bounds__(kBlock) axpy_kernel(AxpyParams p) {
  axpy_op<kBlock>(p, 0, p.n);
}

}  // namespace miso::kernels
