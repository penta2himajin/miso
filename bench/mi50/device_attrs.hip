// Print HIP device attributes relevant to engine design.
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  struct {
    hipDeviceAttribute_t a;
    const char* n;
  } attrs[] = {
      {hipDeviceAttributeVirtualMemoryManagementSupported, "virtual_memory_management"},
      {hipDeviceAttributeCooperativeLaunch, "cooperative_launch"},
      {hipDeviceAttributeManagedMemory, "managed_memory"},
      {hipDeviceAttributePageableMemoryAccess, "pageable_memory_access"},
      {hipDeviceAttributeConcurrentKernels, "concurrent_kernels"},
      {hipDeviceAttributeConcurrentManagedAccess, "concurrent_managed_access"},
      {hipDeviceAttributeMaxRegistersPerBlock, "max_registers_per_block"},
      {hipDeviceAttributeMaxSharedMemoryPerBlock, "max_shared_mem_per_block"},
      {hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, "max_shared_mem_per_cu"},
      {hipDeviceAttributeMaxThreadsPerMultiProcessor, "max_threads_per_cu"},
      {hipDeviceAttributeL2CacheSize, "l2_cache_size_reported"},
      {hipDeviceAttributeWallClockRate, "wall_clock_rate_khz"},
      {hipDeviceAttributeAsyncEngineCount, "async_engine_count"},
      {hipDeviceAttributeHostNativeAtomicSupported, "host_native_atomic"},
      {hipDeviceAttributeMemoryPoolsSupported, "memory_pools_supported"},
  };
  for (auto& x : attrs) {
    int v = -1;
    hipError_t e = hipDeviceGetAttribute(&v, x.a, 0);
    printf("%-28s %d%s\n", x.n, v, e == hipSuccess ? "" : " (query failed)");
  }
  return 0;
}
