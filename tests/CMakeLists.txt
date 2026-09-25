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

set(MISO_ORNITH_GGUF
    "/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf"
    CACHE FILEPATH "Ornith Q4_K_M GGUF; tests that need it are skipped when absent")
set(MISO_TEST_DEFINITIONS
    MISO_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    MISO_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
    MISO_ORNITH_GGUF="${MISO_ORNITH_GGUF}")

miso_gpu_test(test_device test_device.hip)
miso_gpu_test(test_axpy test_axpy.hip)
miso_gpu_test(test_q4k_gemv test_q4k_gemv.hip)
target_compile_definitions(test_q4k_gemv PRIVATE ${MISO_TEST_DEFINITIONS})

# miso_cpu_test(<name> <sources...>): a host-only doctest executable.
function(miso_cpu_test name)
  add_executable(${name} ${ARGN} $<TARGET_OBJECTS:miso_test_main>)
  target_link_libraries(${name} PRIVATE doctest miso_host)
  target_compile_definitions(${name} PRIVATE ${MISO_TEST_DEFINITIONS})
  miso_warnings(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

miso_cpu_test(test_gguf test_gguf.cpp)
miso_cpu_test(test_golden test_golden.cpp)
miso_cpu_test(test_repack test_repack.cpp)
