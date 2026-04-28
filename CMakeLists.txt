cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_O3DEBUG "-ggdb -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Choose build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_ROCM "Whether to use rocm" ON)
option(USE_BNXT "Whether to use BNXT NIC" OFF)
option(USE_IONIC "Whether to use IONIC" OFF)
option(BUILD_EXAMPLES "Whether to build examples" ON)
option(WITH_MPI
       "Enable MPI support (required for mpi_bootstrap and C++ examples)"
       ${BUILD_EXAMPLES})
option(BUILD_APPLICATION "Whether to build application library" ON)
option(BUILD_SHMEM "Whether to build shmem library" ON)
option(BUILD_OPS "Whether to build mori operation kernels" ON)
option(BUILD_IO "Whether to build mori io library" ON)
option(BUILD_COLLECTIVE "Whether to build mori collective library" ON)
option(BUILD_PYBINDS "Whether to build mori python bindings" ON)
option(BUILD_XLA_FFI_OPS "Whether to build mori xla ffi python bindings" OFF)
option(BUILD_UMBP "Whether to build UMBP (Unified Memory/Bandwidth Pool)" OFF)
option(BUILD_TESTS "Whether to build mori CPP tests" ON)
option(BUILD_BENCHMARK "Whether to build benchmark targets" OFF)
option(ENABLE_PROFILER "Enable kernel profiling" OFF)
option(ENABLE_DEBUG_PRINTF "Enable debug printf in device kernels" OFF)
option(ENABLE_STANDARD_MOE_ADAPT "Enable standard moe adapt" OFF)
option(BUILD_OPS_DEVICE "AOT compile EP kernels to .hsaco(requires hipcc)" OFF)

# ---------------------------------------------------------------------------
# Host NIC: all vendor-specific dv libraries (mlx5, bnxt_re, ionic) are loaded
# at runtime via dlopen. No compile-time library or header detection needed.
# ---------------------------------------------------------------------------
message(STATUS "Host NIC: all dv libraries loaded at runtime via dlopen")

if(DEFINED ENV{MORI_WITH_MPI})
  string(TOUPPER "$ENV{MORI_WITH_MPI}" _env_with_mpi)
  if(_env_with_mpi STREQUAL "ON")
    message(STATUS "MPI: enabled via MORI_WITH_MPI env")
    set(WITH_MPI
        ON
        CACHE BOOL "Enable MPI support" FORCE)
  else()
    message(STATUS "MPI: disabled via MORI_WITH_MPI env")
    set(WITH_MPI
        OFF
        CACHE BOOL "Enable MPI support" FORCE)
  endif()
endif()

if(BUILD_XLA_FFI_OPS)
  if(NOT BUILD_OPS_DEVICE)
    message(FATAL_ERROR "BUILD_XLA_FFI_OPS=ON requires BUILD_OPS_DEVICE=ON")
  endif()
  add_compile_definitions(BUILD_XLA_FFI_OPS)
  add_compile_options(-Wno-deprecated-literal-operator)
endif()

# ---------------------------------------------------------------------------
# Device NIC + GPU arch detection (reuse the shared module in cmake/)
# ---------------------------------------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/MoriDetectDevice.cmake)
mori_detect_device_config()
# Bridge: if user didn't set GPU_TARGETS explicitly, use the detected arch
if(NOT DEFINED GPU_TARGETS AND MORI_GPU_ARCH)
  set(GPU_TARGETS
      "${MORI_GPU_ARCH}"
      CACHE STRING "GPU architectures" FORCE)
endif()

message(STATUS "ENABLE_PROFILER = ${ENABLE_PROFILER}")
if(ENABLE_PROFILER)
  add_compile_definitions(ENABLE_PROFILER)
endif()

message(STATUS "ENABLE_DEBUG_PRINTF = ${ENABLE_DEBUG_PRINTF}")
if(ENABLE_DEBUG_PRINTF)
  add_compile_definitions(MORI_ENABLE_DEBUG_PRINTF)
endif()

message(STATUS "ENABLE_STANDARD_MOE_ADAPT = ${ENABLE_STANDARD_MOE_ADAPT}")
if(ENABLE_STANDARD_MOE_ADAPT)
  add_compile_definitions(ENABLE_STANDARD_MOE_ADAPT)
endif()

if(NOT DEFINED WARP_ACCUM_UNROLL)
  set(WARP_ACCUM_UNROLL 1)
endif()
message(STATUS "WARP_ACCUM_UNROLL is set to: ${WARP_ACCUM_UNROLL}")
add_definitions(-DWARP_ACCUM_UNROLL=${WARP_ACCUM_UNROLL})
add_definitions(-DHIP_ENABLE_WARP_SYNC_BUILTINS)
if(USE_ROCM)
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
  project(mori LANGUAGES HIP CXX C)
  find_package(hip REQUIRED)
  link_directories(/opt/rocm/lib)

  if(DEFINED GPU_TARGETS)
    list(GET GPU_TARGETS 0 GPU_ARCH)
    message(STATUS "GPU_TARGETS: ${GPU_TARGETS}")
    # TODO: should iterate all archs
    if(GPU_ARCH MATCHES "^gfx8")
      add_definitions(-D__GFX8__=1)
    elseif(GPU_ARCH MATCHES "^gfx9")
      add_definitions(-D__GFX9__=1)
    endif()
    set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")
  endif()
else()
  set(CUDA_HOME /usr/local/cuda)
  message(STATUS "CUDA_HOME: ${CUDA_HOME}")
  set(CUDAToolkit_Root
      ${CUDA_HOME}
      CACHE PATH "Root of Cuda Toolkit." FORCE)
  set(CMAKE_CUDA_COMPILER
      "${CUDA_HOME}/bin/nvcc"
      CACHE PATH "Root of Cuda Toolkit." FORCE)
  set(CMAKE_CUDA_FLAGS
      "--generate-code=arch=compute_70,code=[sm_70] --generate-code=arch=compute_80,code=[sm_80]"
  )
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

  project(mori LANGUAGES CUDA CXX C)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr")
endif()

set(SPDLOG_BUILD_EXAMPLE
    OFF
    CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH
    OFF
    CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(3rdparty/spdlog)
# Hide spdlog symbols from other libraries to avoid symbol conflicts
target_compile_options(spdlog PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

add_library(mori_logging INTERFACE)
target_include_directories(mori_logging INTERFACE include)
target_link_libraries(mori_logging INTERFACE spdlog::spdlog)

if(ENABLE_PROFILER)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)

  set(PROFILER_GEN_SCRIPT
      ${CMAKE_SOURCE_DIR}/tools/profiler/generate_profiler_bindings.py)
  set(PROFILER_OUTPUT_INCLUDE
      ${CMAKE_BINARY_DIR}/generated/include/mori/profiler)
  set(PROFILER_OUTPUT_PYBIND
      ${CMAKE_BINARY_DIR}/generated/src/pybind/profiler_bindings_generated.cpp)

  file(GLOB_RECURSE KERNEL_SOURCES ${CMAKE_SOURCE_DIR}/src/ops/*.cpp)

  message(STATUS "Generating profiler bindings at configure time...")
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} ${PROFILER_GEN_SCRIPT} ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/src ${PROFILER_OUTPUT_INCLUDE}
      ${PROFILER_OUTPUT_PYBIND}
    RESULT_VARIABLE PROFILER_GEN_RESULT
    OUTPUT_VARIABLE PROFILER_GEN_OUTPUT
    ERROR_VARIABLE PROFILER_GEN_ERROR)

  if(NOT PROFILER_GEN_RESULT EQUAL 0)
    message(
      FATAL_ERROR
        "Failed to generate profiler bindings:\nSTDOUT:\n${PROFILER_GEN_OUTPUT}\nSTDERR:\n${PROFILER_GEN_ERROR}"
    )
  endif()

  add_custom_command(
    OUTPUT ${PROFILER_OUTPUT_PYBIND}
    COMMAND
      ${Python3_EXECUTABLE} ${PROFILER_GEN_SCRIPT} ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/src ${PROFILER_OUTPUT_INCLUDE}
      ${PROFILER_OUTPUT_PYBIND}
    DEPENDS ${KERNEL_SOURCES} ${PROFILER_GEN_SCRIPT}
    COMMENT "Regenerating profiler bindings..."
    VERBATIM)

  add_custom_target(generate_profiler_bindings
                    DEPENDS ${PROFILER_OUTPUT_PYBIND})

  add_library(mori_profiler_interface INTERFACE)
  target_include_directories(mori_profiler_interface
                             INTERFACE ${CMAKE_BINARY_DIR}/generated/include)
  add_dependencies(mori_profiler_interface generate_profiler_bindings)
endif()

if(BUILD_EXAMPLES)
  set(BUILD_APPLICATION ON)
  add_subdirectory(examples)
endif()

if(BUILD_BENCHMARK)
  if(NOT BUILD_APPLICATION)
    message(FATAL_ERROR "BUILD_BENCHMARK=ON requires BUILD_APPLICATION=ON")
  endif()
  if(NOT BUILD_SHMEM)
    message(
      WARNING
        "BUILD_BENCHMARK=ON with BUILD_SHMEM=OFF: add_mori_benchmark_shmem targets will fail until BUILD_SHMEM is enabled"
    )
  endif()
  add_subdirectory(benchmark)
endif()

if(BUILD_APPLICATION)
  add_subdirectory(src/application)
endif()

if(BUILD_SHMEM)
  add_subdirectory(src/shmem)
endif()

if(BUILD_OPS)
  add_subdirectory(src/ops)
endif()

if(BUILD_IO)
  add_subdirectory(src/io)
endif()

if(BUILD_COLLECTIVE)
  add_subdirectory(src/collective)
endif()

if(BUILD_PYBINDS)
  add_subdirectory(src/pybind)
endif()

if(BUILD_UMBP)
  add_subdirectory(src/umbp)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
# Shared libraries
if(BUILD_APPLICATION)
  install(TARGETS mori_application LIBRARY DESTINATION lib)
endif()
if(BUILD_SHMEM)
  install(TARGETS mori_shmem LIBRARY DESTINATION lib)
endif()
if(BUILD_OPS)
  install(TARGETS mori_ops LIBRARY DESTINATION lib)
endif()
if(BUILD_IO)
  install(TARGETS mori_io LIBRARY DESTINATION lib)
endif()

# Public headers
install(DIRECTORY include/mori DESTINATION include)

# CMake package config (enables find_package(mori) for external C++ users)
install(FILES cmake/mori-config.cmake cmake/MoriDetectDevice.cmake
        DESTINATION lib/cmake/mori)

# Logging (spdlog is statically linked, but install the shared lib if built)
if(TARGET spdlog)
  install(TARGETS spdlog ARCHIVE DESTINATION lib)
endif()
