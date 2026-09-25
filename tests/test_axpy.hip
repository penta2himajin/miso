#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <vector>

#include "axpy.hpp"
#include "hip_util.hpp"

namespace {

constexpr int kBlock = 256;

std::vector<float> iota(unsigned n, float scale) {
  std::vector<float> v(n);
  for (unsigned i = 0; i < n; ++i)
    v[i] = scale * static_cast<float>(i % 1024);
  return v;
}

__global__ void __launch_bounds__(kBlock)
    axpy_range_kernel(miso::kernels::AxpyParams p, unsigned first, unsigned count) {
  miso::kernels::axpy_op<kBlock>(p, first, count);
}

}  // namespace

TEST_CASE("axpy_kernel matches the CPU reference for any size and grid") {
  for (unsigned n : {1u, 255u, 256u, 257u, 100003u}) {
    for (unsigned grid : {(n + kBlock - 1) / kBlock, 2u}) {
      CAPTURE(n);
      CAPTURE(grid);
      const auto x = iota(n, 0.5f);
      auto y = iota(n, -0.25f);
      miso::DeviceBuffer<float> dx(x), dy(y);

      miso::kernels::AxpyParams p{3.0f, dx.data(), dy.data(), n};
      miso::kernels::axpy_kernel<kBlock><<<grid, kBlock>>>(p);
      MISO_HIP_CHECK(hipGetLastError());

      const auto got = dy.to_host();
      for (unsigned i = 0; i < n; ++i)
        y[i] = 3.0f * x[i] + y[i];
      CHECK(got == y);
    }
  }
}

TEST_CASE("axpy_op touches only its assigned work-item range") {
  const unsigned n = 4096, first = 1000, count = 777;
  const auto x = iota(n, 1.0f);
  const auto y0 = iota(n, 2.0f);
  miso::DeviceBuffer<float> dx(x), dy(y0);

  miso::kernels::AxpyParams p{-1.0f, dx.data(), dy.data(), n};
  axpy_range_kernel<<<3, kBlock>>>(p, first, count);
  MISO_HIP_CHECK(hipGetLastError());

  const auto got = dy.to_host();
  for (unsigned i = 0; i < n; ++i) {
    const bool inside = i >= first && i < first + count;
    const float want = inside ? -x[i] + y0[i] : y0[i];
    if (got[i] != want) {
      FAIL_CHECK("element " << i << " = " << got[i] << ", want " << want);
    }
  }
}
