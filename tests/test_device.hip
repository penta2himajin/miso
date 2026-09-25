#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <string>
#include <utility>
#include <vector>

#include "hip_util.hpp"

TEST_CASE("the HIP runtime sees an MI50 (gfx906, 60 CUs, wave64)") {
  int count = 0;
  MISO_HIP_CHECK(hipGetDeviceCount(&count));
  REQUIRE(count >= 1);

  hipDeviceProp_t prop{};
  MISO_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  CHECK(std::string(prop.gcnArchName).rfind("gfx906", 0) == 0);
  CHECK(prop.multiProcessorCount == 60);
  CHECK(prop.warpSize == 64);
}

TEST_CASE("DeviceBuffer moves ownership of its allocation") {
  miso::DeviceBuffer<int> a(std::vector<int>{1, 2, 3});
  int* const p = a.data();
  miso::DeviceBuffer<int> b(std::move(a));
  CHECK(b.data() == p);
  CHECK(b.size() == 3);
  CHECK(a.data() == nullptr);
  CHECK(a.size() == 0);
  CHECK(b.to_host() == std::vector<int>{1, 2, 3});
}

TEST_CASE("hip_check turns a HIP error into an exception naming the call") {
  CHECK_THROWS_WITH(MISO_HIP_CHECK(hipErrorInvalidValue),
                    doctest::Contains("hipErrorInvalidValue"));
  CHECK_NOTHROW(MISO_HIP_CHECK(hipSuccess));
}
