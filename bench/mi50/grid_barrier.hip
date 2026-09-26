// Grid-wide barrier cost on gfx906, the prerequisite measurement for the ADR_001 D4 megakernel
// trigger. Probes:
//   A. one cooperative launch executing B back-to-back grid.sync()s, reported per barrier.
//   A'. the same B syncs as B separate non-cooperative launches, i.e. B kernel boundaries.
//   B.  a chain of K dependent steps (each reads the previous step's global write), run as K
//       stream-ordered launches vs K steps inside one cooperative kernel with K-1 barriers.
//       This is the shape a megakernel would replace: the gap B must close is (B1) - (B0).
//
// Cooperative launch needs co-resident blocks, so the grid is capped by the occupancy API and
// the cap is printed.
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <cstdio>

#include "hip_util.hpp"

namespace cg = cooperative_groups;

namespace {

constexpr int kBlock = 256;
constexpr unsigned kN = 256;  // floats per step (small, like a decode intermediate)

// A: B grid barriers and nothing else. `sink` keeps the loop from being optimised away.
__global__ void __launch_bounds__(kBlock) barrier_kernel(unsigned b, unsigned* sink) {
  __shared__ unsigned s;
  if (threadIdx.x == 0)
    s = 0;
  for (unsigned i = 0; i < b; ++i) {
    cg::this_grid().sync();
    if (threadIdx.x == 0)
      s += i;
  }
  if (threadIdx.x == 0 && s == 0xffffffffu)
    atomicAdd(sink, s);
}

// One step body: y = f(x). Cheap, but a global write the next step depends on.
__device__ void step_body(const float* x, float* y, unsigned n) {
  for (unsigned i = threadIdx.x; i < n; i += kBlock)
    y[i] = x[i] * 0.5f + 1.0f;
}

__global__ void __launch_bounds__(kBlock) single_step_kernel(const float* x, float* y, unsigned n) {
  step_body(x, y, n);
}

// A plain (non-cooperative) kernel with the same body as one barrier step, so A' measures a real
// kernel boundary: launch + the dependent write, with no cooperative-launch surcharge.
__global__ void __launch_bounds__(kBlock) plain_boundary_kernel(unsigned* sink) {
  if (threadIdx.x == 0 && sink == nullptr)
    return;
}

// B, fused: K steps in one cooperative launch, ping-ponging two buffers.
__global__ void __launch_bounds__(kBlock)
    chain_kernel(const float* x, float* y, float* z, unsigned n, unsigned steps) {
  for (unsigned i = 0; i < steps; ++i) {
    const float* src = (i % 2 == 0) ? x : y;
    float* dst = (i % 2 == 0) ? y : z;
    step_body(src, dst, n);
    cg::this_grid().sync();
  }
  if (threadIdx.x == 0 && (y == z || n == 0xffffffffu))
    atomicAdd(reinterpret_cast<unsigned*>(z), 0u);
}

}  // namespace

int main() {
  int dev = 0;
  MISO_HIP_CHECK(hipGetDevice(&dev));
  int cus = 0;
  MISO_HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev));
  auto max_blocks = [&](const void* fn) {
    int per_cu = 0;
    MISO_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, fn, kBlock, 0));
    return per_cu * cus;
  };
  const unsigned bmax = static_cast<unsigned>(max_blocks((void*)barrier_kernel));
  const unsigned cmax = static_cast<unsigned>(max_blocks((void*)chain_kernel));
  std::printf("CUs %d, block %d, max co-resident blocks: barrier %u, chain %u\n", cus, kBlock, bmax,
              cmax);

  unsigned* sink = nullptr;
  float *x = nullptr, *y = nullptr, *z = nullptr;
  MISO_HIP_CHECK(hipMalloc(&sink, 4));
  MISO_HIP_CHECK(hipMalloc(&x, kN * 4));
  MISO_HIP_CHECK(hipMalloc(&y, kN * 4));
  MISO_HIP_CHECK(hipMalloc(&z, kN * 4));
  MISO_HIP_CHECK(hipMemset(sink, 0, 4));
  MISO_HIP_CHECK(hipMemset(x, 1, kN * 4));

  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  auto time = [&](int reps, auto&& launch) {
    launch();  // warm up + legality check
    MISO_HIP_CHECK(hipDeviceSynchronize());
    MISO_HIP_CHECK(hipEventRecord(a));
    for (int r = 0; r < reps; ++r)
      launch();
    MISO_HIP_CHECK(hipEventRecord(b));
    MISO_HIP_CHECK(hipEventSynchronize(b));
    float ms;
    MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms * 1e3 / reps;
  };

  const unsigned bars = 20;
  int reps = 50;
  std::printf("\n=== A / A': %u syncs, cooperative vs separate launches ===\n", bars);
  for (unsigned grid : {60u, 120u, 240u, 480u}) {
    if (grid > bmax)
      continue;
    void* args[] = {(void*)&bars, (void*)&sink};
    const float coop = time(reps, [&] {
      MISO_HIP_CHECK(hipLaunchCooperativeKernel((void*)barrier_kernel, dim3(grid), dim3(kBlock),
                                                args, 0, nullptr));
    });
    // A': `bars` separate plain launches, each one kernel boundary.
    const float sep = time(reps, [&] {
      for (unsigned i = 0; i < bars; ++i)
        plain_boundary_kernel<<<grid, kBlock>>>(sink);
    });
    std::printf(
        "grid %4u: coop %6.2f us/launch = %5.3f us/barrier | %2u plain launches %6.2f = "
        "%5.3f us/boundary\n",
        grid, coop, coop / bars, bars, sep, sep / bars);
  }

  std::printf("\n=== B: %u dependent steps over %u floats ===\n", bars, kN);
  {
    const unsigned grid = 60;
    const float sep = time(reps, [&] {
      for (unsigned i = 0; i < bars; ++i)
        single_step_kernel<<<grid, kBlock>>>(x, y, kN);
    });
    std::printf("K separate launches: %6.2f us/chain (%5.3f us/step)\n", sep, sep / bars);
  }
  for (unsigned grid : {60u, 120u, 240u}) {
    if (grid > cmax)
      continue;
    unsigned steps = bars;
    void* args[] = {(void*)&x, (void*)&y, (void*)&z, (void*)&kN, (void*)&steps};
    const float coop = time(reps, [&] {
      MISO_HIP_CHECK(hipLaunchCooperativeKernel((void*)chain_kernel, dim3(grid), dim3(kBlock), args,
                                                0, nullptr));
    });
    std::printf("coop 1 launch grid %4u: %6.2f us/chain (%5.3f us/step)\n", grid, coop,
                coop / bars);
  }

  MISO_HIP_CHECK(hipEventDestroy(a));
  MISO_HIP_CHECK(hipEventDestroy(b));
  MISO_HIP_CHECK(hipFree(sink));
  MISO_HIP_CHECK(hipFree(x));
  MISO_HIP_CHECK(hipFree(y));
  MISO_HIP_CHECK(hipFree(z));
  return 0;
}
