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
option(BUILD_APPLICATION "Whether to build application library" ON)
option(BUILD_SHMEM "Whether to build shmem library" ON)
option(BUILD_OPS "Whether to build mori operation kernels" ON)
option(BUILD_IO "Whether to build mori io library" ON)
option(BUILD_PYBINDS "Whether to build mori python bindings" ON)
option(BUILD_TESTS "Whether to build mori CPP tests" ON)
option(ENABLE_PROFILER "Enable kernel profiling" OFF)
option(ENABLE_DEBUG_PRINTF "Enable debug printf in device kernels" OFF)
option(ENABLE_STANDARD_MOE_ADAPT "Enable standard moe adapt" OFF)

set(CMAKE_LIBRARY_PATH "/usr/local/lib;${CMAKE_LIBRARY_PATH}")
find_library(BNXT_RE_LIB NAMES bnxt_re)
if(BNXT_RE_LIB)
  message(STATUS "Found bnxt_re library at: ${BNXT_RE_LIB}")
else()
  message(
    STATUS "Could NOT find bnxt_re library. BNXT features will be disabled.")
  set(USE_BNXT
      OFF
      CACHE BOOL "Enable BNXT features if bnxt_re library is found" FORCE)
endif()
message(STATUS "USE_BNXT = ${USE_BNXT}")
if(USE_BNXT)
  add_compile_definitions(ENABLE_BNXT)
endif()

find_library(
  IONIC_LIB
  NAMES ionic
  HINTS /lib/x86_64-linux-gnu)
if(IONIC_LIB)
  message(STATUS "Found ionic library at: ${IONIC_LIB}")
else()
  message(
    STATUS "Could NOT find ionic library. IONIC features will be disabled.")
  set(USE_IONIC
      OFF
      CACHE BOOL "Enable IONIC features if ionic library is found" FORCE)
endif()
message(STATUS "USE_IONIC = ${USE_IONIC}")
if(USE_IONIC)
  add_compile_definitions(ENABLE_IONIC)
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
  # set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
  find_package(hip REQUIRED)

  if(DEFINED GPU_TARGETS)
    list(GET GPU_TARGETS 0 GPU_ARCH)
    message(STATUS "GPU_TARGETS: ${GPU_TARGETS}")
    # TODO: should iterate all archs
    if(GPU_ARCH MATCHES "^gfx8")
      add_definitions(-D__GFX8__=1)
    elseif(GPU_ARCH MATCHES "^gfx9")
      add_definitions(-D__GFX9__=1)
    endif()
    set(HIP_HIPCC_FLAGS "--amdgpu-target=${GPU_TARGETS}")
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

add_library(mori_logging INTERFACE)
target_include_directories(mori_logging INTERFACE include
                                                  3rdparty/spdlog/include)

if(ENABLE_PROFILER)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)

  set(PROFILER_GEN_SCRIPT
      ${CMAKE_SOURCE_DIR}/tools/generate_profiler_bindings.py)
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

if(BUILD_PYBINDS)
  add_subdirectory(src/pybind)
endif()

if(BUILD_TESTS)
  add_subdirectory(tests/cpp)
endif()
