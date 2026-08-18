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
option(BUILD_BENCHMARK "Whether to build benchmark targets" OFF)
# MPI is required by mpi_bootstrap, the C++ examples, and every benchmark
# target (all use the MPI multi-rank launcher), so default it on whenever
# either is enabled.
if(BUILD_EXAMPLES OR BUILD_BENCHMARK)
  set(_with_mpi_default ON)
else()
  set(_with_mpi_default OFF)
endif()
option(WITH_MPI
       "Enable MPI support (required for mpi_bootstrap, C++ examples, benchmarks)"
       ${_with_mpi_default})
option(BUILD_APPLICATION "Whether to build application library" ON)
option(BUILD_SHMEM "Whether to build shmem library" ON)
option(BUILD_CCO "Whether to build cco library" ON)

if(BUILD_BENCHMARK)
  set(_cco_sdma_default ON)
else()
  set(_cco_sdma_default OFF)
endif()
option(BUILD_CCO_SDMA "Compile the cco SDMA intra-node copy-engine path"
       ${_cco_sdma_default})
if(BUILD_BENCHMARK AND NOT BUILD_CCO_SDMA)
  message(WARNING
    "BUILD_BENCHMARK=ON requires BUILD_CCO_SDMA=ON (cco benchmarks use ccoSdma); "
    "forcing BUILD_CCO_SDMA=ON.")
  set(BUILD_CCO_SDMA ON CACHE BOOL "Compile the cco SDMA intra-node copy-engine path" FORCE)
endif()

option(BUILD_OPS "Whether to build mori operation kernels" ON)
option(BUILD_IO "Whether to build mori io library" ON)
option(BUILD_COLLECTIVE "Whether to build mori collective library" ON)
option(BUILD_PYBINDS "Whether to build mori python bindings" ON)
option(BUILD_XLA_FFI_OPS "Whether to build mori xla ffi python bindings" OFF)
option(BUILD_UMBP "Whether to build UMBP (Unified Memory/Bandwidth Pool)" OFF)
option(BUILD_METRICS "Whether to build the Prometheus metrics server module" ON)
option(BUILD_TESTS "Whether to build mori CPP tests" ON)
option(ENABLE_PROFILER "Enable kernel profiling" OFF)
option(ENABLE_DEBUG_PRINTF "Enable debug printf in device kernels" OFF)
option(ENABLE_STANDARD_MOE_ADAPT "Enable standard moe adapt" OFF)
option(BUILD_OPS_DEVICE "AOT compile EP kernels to .hsaco(requires hipcc)" OFF)
option(MORI_MULTITHREAD_SUPPORT
       "Enable single-process multi-thread multi-GPU support in mori_shmem" OFF)

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
  # Resolve ROCm before project(... HIP). Not via `hipconfig`: the rocm7.14
  # pytorch image ships a shim at /opt/venv/bin/hipconfig that shadows the real
  # one and always reports the pip rocm-sdk-core wheel dir, which has no CMake
  # configs — CMake then dies with "Failed to find HIP root directory".
  if(NOT DEFINED MORI_ROCM_PATH)
    if(DEFINED ROCM_PATH)
      set(MORI_ROCM_PATH "${ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
      set(MORI_ROCM_PATH "$ENV{ROCM_PATH}")
    elseif(DEFINED ENV{HIP_PATH})
      set(MORI_ROCM_PATH "$ENV{HIP_PATH}")
    else()
      set(MORI_ROCM_PATH "/opt/rocm")
    endif()
  endif()
  if(NOT EXISTS "${MORI_ROCM_PATH}/lib/cmake/hip-lang/hip-lang-config.cmake")
    message(FATAL_ERROR
      "No ROCm dev toolchain at '${MORI_ROCM_PATH}' "
      "(missing lib/cmake/hip-lang/hip-lang-config.cmake).\n"
      "Install the apt ROCm dev stack (see docker/Dockerfile.dev), or point at "
      "an existing install with -DROCM_PATH=<path>.")
  endif()
  message(STATUS "Mori ROCm path: ${MORI_ROCM_PATH}")
  set(ROCM_PATH "${MORI_ROCM_PATH}" CACHE PATH "ROCm root" FORCE)
  # CMakeDetermineHIPCompiler consults this (and only falls back to hipconfig).
  set(CMAKE_HIP_COMPILER_ROCM_ROOT "${MORI_ROCM_PATH}" CACHE PATH "" FORCE)
  # Pin the compiler too: the venv also shims amdclang++ into the wheel.
  if(NOT DEFINED CMAKE_HIP_COMPILER AND EXISTS "${MORI_ROCM_PATH}/llvm/bin/amdclang++")
    set(CMAKE_HIP_COMPILER "${MORI_ROCM_PATH}/llvm/bin/amdclang++" CACHE FILEPATH "" FORCE)
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${MORI_ROCM_PATH}")
  # RPATH the ROCm libdir so the installed libraries resolve libamdhip64 /
  # libhsa-runtime64 without the caller exporting LD_LIBRARY_PATH. Targets that
  # set their own RPATH repeat this (see src/*/CMakeLists.txt).
  list(APPEND CMAKE_INSTALL_RPATH "${MORI_ROCM_PATH}/lib")
  list(APPEND CMAKE_BUILD_RPATH "${MORI_ROCM_PATH}/lib")
  project(mori LANGUAGES HIP CXX C)
  find_package(hip REQUIRED)
  link_directories(${MORI_ROCM_PATH}/lib)

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
# Hide spdlog symbols to avoid cross-library symbol collisions. Both rocroller
# (via hipblaslt/TE) and MORI statically link spdlog; if both export spdlog
# symbols, MORI's spdlog calls can be interposed by rocroller's copy, causing
# heap corruption (free(): invalid size). Using the compiled spdlog library (not
# header-only) with hidden visibility ensures spdlog symbols stay internal to
# MORI's .so files.
set_target_properties(spdlog PROPERTIES CXX_VISIBILITY_PRESET hidden
                                        VISIBILITY_INLINES_HIDDEN ON)

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

if(BUILD_CCO)
  add_subdirectory(src/cco)
endif()

# JIT v2 host runtime. Host-only (HIP runtime API, no device code), so it needs
# no hipcc at build time — see docs/MORI_JIT_V2_DESIGN.md.
add_subdirectory(src/jit/v2)
add_subdirectory(src/ops/dispatch_combine_v2)

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

if(BUILD_METRICS)
  add_subdirectory(src/metrics)
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
if(BUILD_CCO)
  install(TARGETS mori_cco LIBRARY DESTINATION lib)
endif()
if(BUILD_OPS)
  install(TARGETS mori_ops LIBRARY DESTINATION lib)
endif()
if(BUILD_IO)
  install(TARGETS mori_io LIBRARY DESTINATION lib)
endif()
if(BUILD_METRICS)
  install(TARGETS mori_metrics LIBRARY DESTINATION lib)
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
