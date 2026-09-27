#pragma once

// Benchmark-only controls. Empty dispatches deliberately carry no memory dependency.
#include <hip/hip_runtime.h>

#include "rmsnorm.hpp"

namespace miso::norm_bench {

constexpr unsigned kN = 2048, kBlock = 512;
enum class Kind { Empty, Copy, ReadWrite, Norm, Direct };
using Params = kernels::AddRmsNormParams;

__global__ void __launch_bounds__(kBlock) empty_kernel() {}

__global__ void __launch_bounds__(kBlock) copy_kernel(Params p) {
  for (unsigned i = threadIdx.x; i < kN; i += kBlock)
    p.out[i] = p.residual[i];
}

// Same global input/output traffic and residual ownership as Norm, with reduction/rsqrt removed.
__global__ void __launch_bounds__(kBlock) readwrite_kernel(Params p) {
  for (unsigned i = threadIdx.x; i < kN; i += kBlock) {
    const float h = p.delta != nullptr ? p.residual[i] + p.delta[i] : p.residual[i];
    if (p.delta != nullptr)
      p.residual[i] = h;
    p.out[i] = h * p.weight[i];
  }
}

// Both writes also make Empty/Direct chain controls well-defined: consumer reads the producer's
// output directly when the middle operator does no work. Every variant pays these two writes.
__global__ void __launch_bounds__(kBlock)
    producer_kernel(const float* x, float* residual, float* middle) {
  for (unsigned i = threadIdx.x; i < kN; i += kBlock) {
    const float v = __fmaf_rn(x[i], 0.5f, 0.25f);
    residual[i] = v;
    middle[i] = v;
  }
}

__global__ void __launch_bounds__(kBlock) consumer_kernel(const float* x, float* y) {
  for (unsigned i = threadIdx.x; i < kN; i += kBlock)
    y[i] = __fmaf_rn(x[i], 0.5f, -0.125f);
}

inline void launch(Kind kind, Params p, hipStream_t stream) {
  if (kind == Kind::Empty)
    empty_kernel<<<1, kBlock, 0, stream>>>();
  else if (kind == Kind::Copy)
    copy_kernel<<<1, kBlock, 0, stream>>>(p);
  else if (kind == Kind::ReadWrite)
    readwrite_kernel<<<1, kBlock, 0, stream>>>(p);
  else if (kind == Kind::Norm)
    kernels::add_rmsnorm_kernel<kN, kBlock><<<1, kBlock, 0, stream>>>(p, 1);
}

}  // namespace miso::norm_bench
