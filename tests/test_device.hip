#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <string>

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

TEST_CASE("hip_check turns a HIP error into an exception naming the call") {
  CHECK_THROWS_WITH(MISO_HIP_CHECK(hipErrorInvalidValue),
                    doctest::Contains("hipErrorInvalidValue"));
  CHECK_NOTHROW(MISO_HIP_CHECK(hipSuccess));
}
