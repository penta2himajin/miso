# miso_bench(<name> <sources...>): benchmark executable (not run by CTest).
function(miso_bench name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE miso_kernels miso_host)
  target_compile_definitions(${name} PRIVATE
    MISO_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
    MISO_ORNITH_GGUF="${MISO_ORNITH_GGUF}")
  miso_warnings(${name})
  miso_kernel_report(${name})
endfunction()

miso_bench(bench_q4k_gemv gemv/bench_q4k_gemv.hip)
