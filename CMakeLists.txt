cmake_minimum_required(VERSION 3.21)

set(CMAKE_HIP_ARCHITECTURES gfx906 CACHE STRING "AMDGPU target")

# clang selects the newest GCC under /usr/lib/gcc. When that GCC's libstdc++ headers are not
# installed, every HIP compile fails with "'cmath' file not found", so pick the newest GCC that
# has them.
set(MISO_GCC_INSTALL_DIR "" CACHE PATH "GCC install dir passed to clang (auto-detected if empty)")
if(NOT MISO_GCC_INSTALL_DIR)
  file(GLOB _gcc_dirs LIST_DIRECTORIES true /usr/lib/gcc/x86_64-linux-gnu/*)
  list(SORT _gcc_dirs COMPARE NATURAL ORDER DESCENDING)
  list(GET _gcc_dirs 0 _gcc_newest)
  get_filename_component(_gcc_ver "${_gcc_newest}" NAME)
  if(NOT EXISTS "/usr/include/c++/${_gcc_ver}")
    foreach(_dir IN LISTS _gcc_dirs)
      get_filename_component(_ver "${_dir}" NAME)
      if(EXISTS "/usr/include/c++/${_ver}")
        set(MISO_GCC_INSTALL_DIR "${_dir}" CACHE PATH "" FORCE)
        break()
      endif()
    endforeach()
  endif()
endif()
if(MISO_GCC_INSTALL_DIR)
  set(CMAKE_HIP_FLAGS_INIT "--gcc-install-dir=${MISO_GCC_INSTALL_DIR}")
endif()

project(miso LANGUAGES CXX HIP)

# CMake 3.22's Makefile generator does not track header dependencies of HIP sources, so edited
# headers silently leave stale objects. Ninja uses compiler depfiles.
if(NOT CMAKE_GENERATOR MATCHES "Ninja")
  message(FATAL_ERROR "Use the Ninja generator (cmake --preset default); got '${CMAKE_GENERATOR}'")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Project code builds warning-free; vendored third_party/ is included as SYSTEM and exempt.
function(miso_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
endfunction()

# Print and record VGPR / SGPR / LDS / scratch / occupancy of every kernel in a built target
# (ADR_001, "D4 -> megakernel"). Output: ${CMAKE_BINARY_DIR}/kernel-resources/<target>.txt
function(miso_kernel_report target)
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/kernel_resources.py
            $<TARGET_FILE:${target}>
            --arch ${CMAKE_HIP_ARCHITECTURES}
            --out ${CMAKE_BINARY_DIR}/kernel-resources/${target}.txt
    VERBATIM)
endfunction()

# Device code: must not see host-only headers from src/.
add_library(miso_kernels INTERFACE)
target_include_directories(miso_kernels INTERFACE ${PROJECT_SOURCE_DIR}/kernels)

# Host code.
add_library(miso_host STATIC src/gguf.cpp src/dequant.cpp src/repack.cpp)
target_include_directories(miso_host PUBLIC ${PROJECT_SOURCE_DIR}/src)
# gguf-py rounds mul and sub separately. For Q4_K/Q6_K the products are exact, so contraction
# does not change results today, but it would for formats with inexact products.
target_compile_options(miso_host PRIVATE -ffp-contract=off)
miso_warnings(miso_host)

add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party)

enable_testing()
add_subdirectory(tests)
add_subdirectory(bench)
