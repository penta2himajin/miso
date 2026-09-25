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

set(MISO_TEST_DEFINITIONS
    MISO_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    MISO_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
    MISO_ORNITH_GGUF="${MISO_ORNITH_GGUF}")

miso_gpu_test(test_device test_device.hip)
miso_gpu_test(test_axpy test_axpy.hip)
miso_gpu_test(test_q4k_gemv test_q4k_gemv.hip)
target_compile_definitions(test_q4k_gemv PRIVATE ${MISO_TEST_DEFINITIONS})
miso_gpu_test(test_q6k_gemv test_q6k_gemv.hip)
target_compile_definitions(test_q6k_gemv PRIVATE ${MISO_TEST_DEFINITIONS})
miso_gpu_test(test_deltanet test_deltanet.hip)
target_compile_definitions(test_deltanet PRIVATE ${MISO_TEST_DEFINITIONS})
target_link_libraries(test_deltanet PRIVATE miso_engine)
miso_gpu_test(test_model test_model.hip)
target_compile_definitions(test_model PRIVATE ${MISO_TEST_DEFINITIONS})
target_link_libraries(test_model PRIVATE miso_engine)
miso_gpu_test(test_moe test_moe.hip)
target_compile_definitions(test_moe PRIVATE ${MISO_TEST_DEFINITIONS})
target_link_libraries(test_moe PRIVATE miso_engine)
miso_gpu_test(test_attn_splitk test_attn_splitk.hip)
miso_gpu_test(test_attention test_attention.hip)
target_compile_definitions(test_attention PRIVATE ${MISO_TEST_DEFINITIONS})
target_link_libraries(test_attention PRIVATE miso_engine)
miso_gpu_test(test_norm_embed test_norm_embed.hip)
target_compile_definitions(test_norm_embed PRIVATE ${MISO_TEST_DEFINITIONS})

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
miso_cpu_test(test_tokenizer test_tokenizer.cpp)

# CLI smoke test: the golden prompt, raw, greedy for 8 tokens, prints llama.cpp's continuation.
if(EXISTS "${MISO_ORNITH_GGUF}")
  add_test(NAME cli_golden_prompt
           COMMAND miso --raw -n 8 "The capital of France is Paris. The capital of Japan is Tokyo. 日本の首都は")
  set_tests_properties(cli_golden_prompt PROPERTIES PASS_REGULAR_EXPRESSION "東京です。 The capital of Japan is")
endif()
