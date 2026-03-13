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
option(BUILD_UMBP "Whether to build UMBP (Unified Memory/Bandwidth Pool)" OFF)
option(BUILD_TESTS "Whether to build mori CPP tests" ON)
option(ENABLE_PROFILER "Enable kernel profiling" OFF)
option(ENABLE_DEBUG_PRINTF "Enable debug printf in device kernels" OFF)
option(ENABLE_STANDARD_MOE_ADAPT "Enable standard moe adapt" OFF)
option(ENABLE_XLA_FFI "Build XLA FFI custom call handlers for JAX integration" OFF)

# ---------------------------------------------------------------------------
# Host NIC: link all available NIC libraries (can be multiple). Only requires
# the library to exist — no device needed (supports CI builds).
# ---------------------------------------------------------------------------
set(CMAKE_LIBRARY_PATH "/usr/local/lib;${CMAKE_LIBRARY_PATH}")

find_library(BNXT_RE_LIB NAMES bnxt_re)
find_library(
  IONIC_LIB
  NAMES ionic
  HINTS /lib/x86_64-linux-gnu)
find_library(MLX5_LIB NAMES mlx5)

find_path(
  BNXT_RE_DV_INCLUDE
  NAMES infiniband/bnxt_re_dv.h
  HINTS /usr/include /usr/local/include)
find_path(
  BNXT_RE_HSI_INCLUDE
  NAMES infiniband/bnxt_re_hsi.h
  HINTS /usr/include /usr/local/include)
if(BNXT_RE_DV_INCLUDE AND BNXT_RE_HSI_INCLUDE)
  set(_bnxt_headers_found TRUE)
else()
  set(_bnxt_headers_found FALSE)
endif()

# USE_BNXT: env override > cmake -D > auto-detect from library + headers
if(DEFINED ENV{USE_BNXT})
  string(TOUPPER "$ENV{USE_BNXT}" _env_bnxt)
  if(_env_bnxt STREQUAL "ON")
    if(BNXT_RE_LIB AND _bnxt_headers_found)
      message(
        STATUS "Host NIC BNXT: enabled via USE_BNXT env (lib=${BNXT_RE_LIB})")
      set(USE_BNXT
          ON
          CACHE BOOL "Enable BNXT host features" FORCE)
    else()
      message(
        WARNING "USE_BNXT=ON but libbnxt_re or headers not found — disabling")
      set(USE_BNXT
          OFF
          CACHE BOOL "Enable BNXT host features" FORCE)
    endif()
  else()
    message(STATUS "Host NIC BNXT: explicitly disabled via USE_BNXT env")
    set(USE_BNXT
        OFF
        CACHE BOOL "Enable BNXT host features" FORCE)
  endif()
elseif(BNXT_RE_LIB AND _bnxt_headers_found)
  message(STATUS "Host NIC BNXT: auto-detected (lib=${BNXT_RE_LIB})")
  set(USE_BNXT
      ON
      CACHE BOOL "Enable BNXT host features" FORCE)
else()
  if(BNXT_RE_LIB AND NOT _bnxt_headers_found)
    message(
      STATUS "Host NIC BNXT: library found but missing headers — disabled")
  else()
    message(STATUS "Host NIC BNXT: not found")
  endif()
  set(USE_BNXT
      OFF
      CACHE BOOL "Enable BNXT host features" FORCE)
endif()
if(USE_BNXT)
  add_compile_definitions(ENABLE_BNXT)
endif()

# USE_IONIC: env override > cmake -D > auto-detect from library
if(DEFINED ENV{USE_IONIC})
  string(TOUPPER "$ENV{USE_IONIC}" _env_ionic)
  if(_env_ionic STREQUAL "ON")
    if(IONIC_LIB)
      message(
        STATUS "Host NIC IONIC: enabled via USE_IONIC env (lib=${IONIC_LIB})")
      set(USE_IONIC
          ON
          CACHE BOOL "Enable IONIC host features" FORCE)
    else()
      message(WARNING "USE_IONIC=ON but libionic not found — disabling")
      set(USE_IONIC
          OFF
          CACHE BOOL "Enable IONIC host features" FORCE)
    endif()
  else()
    message(STATUS "Host NIC IONIC: explicitly disabled via USE_IONIC env")
    set(USE_IONIC
        OFF
        CACHE BOOL "Enable IONIC host features" FORCE)
  endif()
elseif(IONIC_LIB)
  message(STATUS "Host NIC IONIC: auto-detected (lib=${IONIC_LIB})")
  set(USE_IONIC
      ON
      CACHE BOOL "Enable IONIC host features" FORCE)
else()
  message(STATUS "Host NIC IONIC: not found")
  set(USE_IONIC
      OFF
      CACHE BOOL "Enable IONIC host features" FORCE)
endif()
if(USE_IONIC)
  add_compile_definitions(ENABLE_IONIC)
endif()
message(STATUS "Host NIC: USE_BNXT=${USE_BNXT}, USE_IONIC=${USE_IONIC}")

# ---------------------------------------------------------------------------
# Device NIC: detect exactly one provider for IBGDA dispatch in device code.
# Same logic as Python JIT detect_nic_type(): env > sysfs > lspci > host libs.
# Sets MORI_DEVICE_NIC_DEFINE to the compile definition for device targets.
# ---------------------------------------------------------------------------
function(detect_device_nic out_var)
  # 1. Env override
  if(DEFINED ENV{MORI_DEVICE_NIC})
    string(TOLOWER "$ENV{MORI_DEVICE_NIC}" _nic)
    message(
      STATUS "Device NIC: override from MORI_DEVICE_NIC=$ENV{MORI_DEVICE_NIC}")
    set(${out_var}
        "${_nic}"
        PARENT_SCOPE)
    return()
  endif()

  # 1. /sys/class/infiniband/ — pick NIC with most devices + verify library
  file(GLOB _ib_devices "/sys/class/infiniband/*")
  set(_bnxt 0)
  set(_ionic 0)
  set(_mlx5 0)
  foreach(_dev ${_ib_devices})
    get_filename_component(_name ${_dev} NAME)
    if(_name MATCHES "^bnxt_re")
      math(EXPR _bnxt "${_bnxt} + 1")
    elseif(_name MATCHES "^ionic")
      math(EXPR _ionic "${_ionic} + 1")
    elseif(_name MATCHES "^mlx5")
      math(EXPR _mlx5 "${_mlx5} + 1")
    else()
      execute_process(
        COMMAND readlink -f "${_dev}/device/driver"
        OUTPUT_VARIABLE _drv
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        RESULT_VARIABLE _rc)
      if(_rc EQUAL 0)
        get_filename_component(_drv_name "${_drv}" NAME)
        if(_drv_name MATCHES "^bnxt")
          math(EXPR _bnxt "${_bnxt} + 1")
        elseif(_drv_name MATCHES "^ionic")
          math(EXPR _ionic "${_ionic} + 1")
        elseif(_drv_name MATCHES "^mlx5")
          math(EXPR _mlx5 "${_mlx5} + 1")
        endif()
      endif()
    endif()
  endforeach()

  # Pick the NIC with the most devices, verify library exists. Tie-break: mlx5 >
  # bnxt > ionic (mlx5 inserted first = stable sort priority). Use ":" separator
  # since ";" is CMake's list delimiter.
  set(_sysfs_candidates "")
  if(_mlx5 GREATER 0)
    list(APPEND _sysfs_candidates "${_mlx5}:mlx5")
  endif()
  if(_bnxt GREATER 0)
    list(APPEND _sysfs_candidates "${_bnxt}:bnxt")
  endif()
  if(_ionic GREATER 0)
    list(APPEND _sysfs_candidates "${_ionic}:ionic")
  endif()
  if(_sysfs_candidates)
    list(
      SORT _sysfs_candidates
      COMPARE NATURAL
      ORDER DESCENDING)
    message(
      STATUS
        "Device NIC: sysfs counts (mlx5=${_mlx5}, bnxt=${_bnxt}, ionic=${_ionic})"
    )
    foreach(_entry ${_sysfs_candidates})
      string(REGEX REPLACE "^[0-9]+:" "" _nic "${_entry}")
      if((_nic STREQUAL "bnxt"
          AND BNXT_RE_LIB
          AND _bnxt_headers_found)
         OR (_nic STREQUAL "ionic" AND IONIC_LIB)
         OR (_nic STREQUAL "mlx5" AND MLX5_LIB))
        message(
          STATUS "Device NIC: selected '${_nic}' (most devices with library)")
        set(${out_var}
            "${_nic}"
            PARENT_SCOPE)
        return()
      else()
        message(
          STATUS "Device NIC: skipping '${_nic}' (missing library or headers)")
      endif()
    endforeach()
  endif()

  # 1. lspci PCI vendor ID — pick vendor with most devices + verify library
  execute_process(
    COMMAND lspci -nn -d ::0200
    OUTPUT_VARIABLE _lspci
    ERROR_QUIET
    RESULT_VARIABLE _rc)
  if(_rc EQUAL 0 AND _lspci)
    string(REGEX MATCHALL "14e4" _b "${_lspci}")
    string(REGEX MATCHALL "1dd8" _i "${_lspci}")
    string(REGEX MATCHALL "15b3" _m "${_lspci}")
    list(LENGTH _b _bp)
    list(LENGTH _i _ip)
    list(LENGTH _m _mp)

    set(_lspci_candidates "")
    if(_mp GREATER 0)
      list(APPEND _lspci_candidates "${_mp}:mlx5")
    endif()
    if(_bp GREATER 0)
      list(APPEND _lspci_candidates "${_bp}:bnxt")
    endif()
    if(_ip GREATER 0)
      list(APPEND _lspci_candidates "${_ip}:ionic")
    endif()
    if(_lspci_candidates)
      list(
        SORT _lspci_candidates
        COMPARE NATURAL
        ORDER DESCENDING)
      message(
        STATUS
          "Device NIC: lspci counts (mlx5=${_mp}, bnxt=${_bp}, ionic=${_ip})")
      foreach(_entry ${_lspci_candidates})
        string(REGEX REPLACE "^[0-9]+:" "" _nic "${_entry}")
        if((_nic STREQUAL "bnxt"
            AND BNXT_RE_LIB
            AND _bnxt_headers_found)
           OR (_nic STREQUAL "ionic" AND IONIC_LIB)
           OR (_nic STREQUAL "mlx5" AND MLX5_LIB))
          message(STATUS "Device NIC: selected '${_nic}' (lspci + library)")
          set(${out_var}
              "${_nic}"
              PARENT_SCOPE)
          return()
        else()
          message(
            STATUS "Device NIC: skipping '${_nic}' (missing library or headers)"
          )
        endif()
      endforeach()
    endif()
  endif()

  # 1. Fallback: first available library (mlx5 preferred)
  if(MLX5_LIB)
    set(_nic "mlx5")
  elseif(BNXT_RE_LIB AND _bnxt_headers_found)
    set(_nic "bnxt")
  elseif(IONIC_LIB)
    set(_nic "ionic")
  else()
    set(_nic "mlx5")
  endif()
  message(STATUS "Device NIC: fallback to '${_nic}' (library detection)")
  set(${out_var}
      "${_nic}"
      PARENT_SCOPE)
endfunction()

detect_device_nic(_detected_device_nic)
if(_detected_device_nic STREQUAL "bnxt")
  set(MORI_DEVICE_NIC_DEFINE
      "MORI_DEVICE_NIC_BNXT"
      CACHE STRING "Device NIC provider" FORCE)
elseif(_detected_device_nic STREQUAL "ionic")
  set(MORI_DEVICE_NIC_DEFINE
      "MORI_DEVICE_NIC_IONIC"
      CACHE STRING "Device NIC provider" FORCE)
else()
  set(MORI_DEVICE_NIC_DEFINE
      ""
      CACHE STRING "Device NIC provider" FORCE)
endif()
message(
  STATUS "Device NIC: ${_detected_device_nic} (${MORI_DEVICE_NIC_DEFINE})")

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

add_library(mori_logging INTERFACE)
target_include_directories(mori_logging INTERFACE include)
target_link_libraries(mori_logging INTERFACE spdlog::spdlog)

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

if(BUILD_UMBP)
  add_subdirectory(src/umbp)
endif()

if(BUILD_PYBINDS)
  add_subdirectory(src/pybind)
endif()

if(ENABLE_XLA_FFI)
  message(STATUS "ENABLE_XLA_FFI = ON")
  add_subdirectory(src/ffi)
endif()

if(BUILD_TESTS)
  add_subdirectory(tests/cpp)
endif()
