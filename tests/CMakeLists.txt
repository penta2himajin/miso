add_library(miso_test_main OBJECT test_main.cpp)
target_link_libraries(miso_test_main PRIVATE doctest)

# miso_gpu_test(<name> <sources...>): a doctest executable run by CTest on the GPU.
function(miso_gpu_test name)
  add_executable(${name} ${ARGN} $<TARGET_OBJECTS:miso_test_main>)
  target_link_libraries(${name} PRIVATE doctest miso_kernels miso_host)
  miso_warnings(${name})
  miso_kernel_report(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

miso_gpu_test(test_device test_device.hip)
miso_gpu_test(test_axpy test_axpy.hip)
