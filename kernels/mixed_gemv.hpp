#pragma once

// Independent Q6_K / Q4_K projections of the same activation vector. Whole workgroups choose
// one original GEMV operator, with the original logical grid and per-row arithmetic. Interleaving
// the two kinds of workgroup gives the scheduler an opportunity to hide their memory latency.

#include "q4k_gemv.hpp"
#include "q6k_gemv.hpp"

namespace miso::kernels {

template <int Q6Rows, unsigned Q6Grid, unsigned Q4Grid, int kBlock = 256>
__device__ void mixed_gemv_op(const Q6kGemvParams& q6, const Q4kGemvParams& q4) {
  static_assert(Q4Grid > 0 && Q6Grid >= Q4Grid && Q6Grid % Q4Grid == 0);
  constexpr unsigned kQ6PerStripe = Q6Grid / Q4Grid;
  constexpr unsigned kStripe = kQ6PerStripe + 1;
  const unsigned group = blockIdx.x / kStripe;
  const unsigned kind = blockIdx.x % kStripe;
  // The branch is uniform across a workgroup, so all lanes run the same wave reduction.
  if (kind < kQ6PerStripe) {
    q6k_gemv_op<2048, kBlock, Q6Rows>(q6, 0, q6.n_rows, group * kQ6PerStripe + kind, Q6Grid);
  } else {
    q4k_gemv_op<ActFormat::Fp16, 2048, kBlock, 4>(q4, 0, q4.n_rows, group, Q4Grid);
  }
}

template <int Q6Rows, unsigned Q6Grid, unsigned Q4Grid, int kBlock = 256>
__global__ void __launch_bounds__(kBlock) mixed_gemv_kernel(Q6kGemvParams q6, Q4kGemvParams q4) {
  mixed_gemv_op<Q6Rows, Q6Grid, Q4Grid, kBlock>(q6, q4);
}

}  // namespace miso::kernels
