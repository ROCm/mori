# ---------------------------------------------------------------------------
# SPDK auto-detection: try pkg-config, enable if found. Search order: 1.
# SPDK_PKG_CONFIG_PATH (user override) 2. System pkg-config paths (e.g.
# /usr/local/lib/pkgconfig) 3. 3rdparty/spdk/build/lib/pkgconfig (local
# submodule build) Set USE_SPDK=OFF to force-disable even if SPDK is installed.
# ---------------------------------------------------------------------------
set(SPDK_PKG_CONFIG_PATH
    ""
    CACHE STRING "Custom pkg-config path for SPDK")
option(USE_SPDK "Enable SPDK support (auto-detected by default)" ON)

set(HAVE_SPDK FALSE)
if(USE_SPDK)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    # Save and restore PKG_CONFIG_PATH so we don't leak changes.
    set(_spdk_orig_pc_path "$ENV{PKG_CONFIG_PATH}")

    if(SPDK_PKG_CONFIG_PATH)
      set(ENV{PKG_CONFIG_PATH} "${SPDK_PKG_CONFIG_PATH}:${_spdk_orig_pc_path}")
    endif()

    # Try 1: system / user-specified pkg-config paths
    execute_process(
      COMMAND pkg-config --cflags spdk_event spdk_event_bdev spdk_env_dpdk
      OUTPUT_VARIABLE SPDK_CFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE SPDK_CFLAGS_RC
      ERROR_QUIET)
    execute_process(
      COMMAND pkg-config --libs --static spdk_event spdk_event_bdev
              spdk_env_dpdk
      OUTPUT_VARIABLE SPDK_LDFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE SPDK_LDFLAGS_RC
      ERROR_QUIET)

    if(SPDK_CFLAGS_RC EQUAL 0 AND SPDK_LDFLAGS_RC EQUAL 0)
      set(HAVE_SPDK TRUE)
    endif()

    # Try 2: fallback to 3rdparty/spdk local build (from tools/setup_spdk.sh)
    if(NOT HAVE_SPDK)
      set(_spdk_3rdparty_pc
          "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/spdk/build/lib/pkgconfig")
      get_filename_component(_spdk_3rdparty_pc "${_spdk_3rdparty_pc}" ABSOLUTE)
      if(EXISTS "${_spdk_3rdparty_pc}/spdk_event.pc")
        set(ENV{PKG_CONFIG_PATH} "${_spdk_3rdparty_pc}:${_spdk_orig_pc_path}")
        execute_process(
          COMMAND pkg-config --cflags spdk_event spdk_event_bdev spdk_env_dpdk
          OUTPUT_VARIABLE SPDK_CFLAGS
          OUTPUT_STRIP_TRAILING_WHITESPACE
          RESULT_VARIABLE SPDK_CFLAGS_RC
          ERROR_QUIET)
        execute_process(
          COMMAND pkg-config --libs --static spdk_event spdk_event_bdev
                  spdk_env_dpdk
          OUTPUT_VARIABLE SPDK_LDFLAGS
          OUTPUT_STRIP_TRAILING_WHITESPACE
          RESULT_VARIABLE SPDK_LDFLAGS_RC
          ERROR_QUIET)
        if(SPDK_CFLAGS_RC EQUAL 0 AND SPDK_LDFLAGS_RC EQUAL 0)
          set(HAVE_SPDK TRUE)
          message(STATUS "[UMBP] SPDK found via 3rdparty/spdk local build")
        endif()
      endif()
    endif()

    set(ENV{PKG_CONFIG_PATH} "${_spdk_orig_pc_path}")
  endif()

  if(HAVE_SPDK)
    message(STATUS "[UMBP] SPDK detected via pkg-config — SPDK support ENABLED")
    message(STATUS "[UMBP] SPDK CFLAGS: ${SPDK_CFLAGS}")
    message(STATUS "[UMBP] SPDK LDFLAGS: ${SPDK_LDFLAGS}")
  else()
    message(STATUS "[UMBP] SPDK not found — SPDK support DISABLED (POSIX only)")
    message(
      STATUS "[UMBP]   To enable: run tools/setup_spdk.sh, then reconfigure.")
  endif()
endif()

# Core static library
set(UMBP_CORE_SRCS
    local/block_index/local_block_index.cpp
    local/storage/io/storage_io_driver.cpp
    local/storage/io/posix_storage_io_driver.cpp
    local/storage/io/io_uring_storage_io_driver.cpp
    local/storage/copy_pipeline.cpp
    local/storage/segment/segment_format.cpp
    local/storage/segment/segment_index.cpp
    local/storage/segment/segment_scanner.cpp
    local/storage/segment/segment_writer.cpp
    local/storage/tier_backend.cpp
    local/storage/dram_tier.cpp
    local/storage/ssd_tier.cpp
    local/storage/dummy_ssd_tier.cpp
    local/storage/local_storage_manager.cpp
    local/umbp_client.cpp
    allocator/offset_allocator.cpp)

# SPDK Proxy client (shared memory IPC, no SPDK dependency — Linux only)
if(UNIX)
  list(APPEND UMBP_CORE_SRCS proxy/spdk_proxy_shm.cpp
       local/storage/spdk_proxy_tier.cpp)
endif()

if(HAVE_SPDK)
  list(APPEND UMBP_CORE_SRCS local/storage/spdk/spdk_env.cpp
       local/storage/spdk_ssd_tier.cpp)
endif()

add_library(umbp_core STATIC ${UMBP_CORE_SRCS})
target_include_directories(umbp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(umbp_core PRIVATE pthread)
if(UNIX)
  target_link_libraries(umbp_core PRIVATE rt)
endif()

if(HAVE_SPDK)
  target_compile_definitions(umbp_core PUBLIC USE_SPDK)
  separate_arguments(SPDK_CFLAGS_LIST UNIX_COMMAND "${SPDK_CFLAGS}")
  target_compile_options(umbp_core PUBLIC ${SPDK_CFLAGS_LIST})

  separate_arguments(SPDK_LDFLAGS_LIST UNIX_COMMAND "${SPDK_LDFLAGS}")
  target_link_libraries(
    umbp_core
    PUBLIC -Wl,--whole-archive
           ${SPDK_LDFLAGS_LIST}
           -Wl,--no-whole-archive
           -lisal
           -lisal_crypto
           -lssl
           -lcrypto
           -lfuse3
           -lnuma
           -ldl
           -lrt
           -luuid
           -lm
           -laio
           -lpthread
           -lelf)

  # SPDK Proxy daemon (owns SPDK, serves ranks via SHM IPC)
  add_executable(spdk_proxy proxy/spdk_proxy_daemon.cpp)
  target_link_libraries(spdk_proxy PRIVATE umbp_core)
endif()

# Python bindings are registered in src/pybind/pybind_umbp.cpp (compiled into
# libmori_pybinds when BUILD_UMBP=ON)

# =============================================================================
# UMBP Master Control Plane — CMake Build (requires gRPC + Protobuf)
# =============================================================================

find_package(Protobuf QUIET)
find_package(gRPC CONFIG QUIET)

# This branch wires distributed PoolClient support directly into umbp_core, so
# gRPC + Protobuf are a required build dependency here.
if(TARGET gRPC::grpc++)
  set(_GRPCPP_LIB gRPC::grpc++)
elseif(TARGET grpc++)
  set(_GRPCPP_LIB grpc++)
else()
  find_path(GRPC_INCLUDE_DIR grpcpp/grpcpp.h REQUIRED)
  find_library(GRPCPP_LIBRARY NAMES grpc++ REQUIRED)
  find_library(GRPC_LIBRARY NAMES grpc REQUIRED)
  find_library(GPR_LIBRARY NAMES gpr REQUIRED)

  add_library(gRPC::grpc++ UNKNOWN IMPORTED)
  set_target_properties(
    gRPC::grpc++
    PROPERTIES IMPORTED_LOCATION "${GRPCPP_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${GRPC_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${GRPC_LIBRARY};${GPR_LIBRARY}")
  set(_GRPCPP_LIB gRPC::grpc++)
endif()

find_library(_PROTOBUF_LIB NAMES protobuf libprotobuf REQUIRED)
get_filename_component(_PROTOBUF_LIB_DIR "${_PROTOBUF_LIB}" DIRECTORY)
get_filename_component(_PROTOBUF_PREFIX "${_PROTOBUF_LIB_DIR}" DIRECTORY)
find_path(_PROTOBUF_INCLUDE_DIR google/protobuf/message.h
          HINTS "${_PROTOBUF_PREFIX}/include" REQUIRED)
set(_PROTOBUF_LIBS "${_PROTOBUF_LIB}")
message(STATUS "UMBP protobuf lib: ${_PROTOBUF_LIBS}")
message(STATUS "UMBP protobuf include: ${_PROTOBUF_INCLUDE_DIR}")

set(_PROTOBUF_PROTOC "${Protobuf_PROTOC_EXECUTABLE}")
if(NOT _PROTOBUF_PROTOC)
  set(_PROTOBUF_PROTOC "${PROTOBUF_PROTOC_EXECUTABLE}")
endif()
if(NOT _PROTOBUF_PROTOC)
  find_program(_PROTOBUF_PROTOC protoc HINTS "${_PROTOBUF_PREFIX}/bin" REQUIRED)
endif()
message(STATUS "UMBP protoc: ${_PROTOBUF_PROTOC}")

if(TARGET gRPC::grpc_cpp_plugin)
  set(_GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
elseif(TARGET grpc_cpp_plugin)
  set(_GRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)
else()
  find_program(_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
endif()

execute_process(
  COMMAND ${_PROTOBUF_PROTOC} --version
  OUTPUT_VARIABLE _PROTOC_VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
  RESULT_VARIABLE _PROTOC_RC)
if(NOT _PROTOC_RC EQUAL 0)
  message(FATAL_ERROR "[UMBP] protoc failed to run.")
endif()

string(REGEX MATCH "[0-9]+\\.[0-9]+" _PROTOC_MM "${_PROTOC_VERSION_OUTPUT}")
set(_PROTOBUF_MM "")
if(EXISTS "${_PROTOBUF_INCLUDE_DIR}/google/protobuf/stubs/common.h")
  file(STRINGS "${_PROTOBUF_INCLUDE_DIR}/google/protobuf/stubs/common.h"
       _PB_VER_LINE REGEX "^#define GOOGLE_PROTOBUF_VERSION [0-9]+")
  if(_PB_VER_LINE)
    string(REGEX MATCH "[0-9]+" _PB_VER_NUM "${_PB_VER_LINE}")
    math(EXPR _PB_MAJOR "${_PB_VER_NUM} / 1000000")
    math(EXPR _PB_MINOR "(${_PB_VER_NUM} / 1000) % 1000")
    set(_PROTOBUF_MM "${_PB_MAJOR}.${_PB_MINOR}")
  endif()
endif()
if(NOT _PROTOBUF_MM AND DEFINED Protobuf_VERSION)
  string(REGEX MATCH "^[0-9]+\\.[0-9]+" _PROTOBUF_MM "${Protobuf_VERSION}")
endif()
message(
  STATUS
    "[UMBP] protoc version: ${_PROTOC_VERSION_OUTPUT} (${_PROTOC_MM}), libprotobuf headers: ${_PROTOBUF_MM}"
)
if(_PROTOC_MM
   AND _PROTOBUF_MM
   AND NOT _PROTOC_MM STREQUAL _PROTOBUF_MM)
  message(
    FATAL_ERROR
      "[UMBP] protoc ${_PROTOC_MM} vs libprotobuf ${_PROTOBUF_MM} mismatch.")
elseif(NOT _PROTOBUF_MM)
  message(FATAL_ERROR "[UMBP] cannot determine libprotobuf header version.")
endif()

message(
  STATUS "[UMBP] gRPC + Protobuf found — distributed control plane ENABLED")

# ---------- Proto compilation -------------------------------------------------

set(UMBP_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/distributed/proto")
set(UMBP_PROTO_FILE "${UMBP_PROTO_DIR}/umbp.proto")
set(UMBP_PEER_PROTO_FILE "${UMBP_PROTO_DIR}/umbp_peer.proto")
set(UMBP_PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")
file(MAKE_DIRECTORY ${UMBP_PROTO_GEN_DIR})

set(UMBP_PROTO_SRCS "${UMBP_PROTO_GEN_DIR}/umbp.pb.cc")
set(UMBP_PROTO_HDRS "${UMBP_PROTO_GEN_DIR}/umbp.pb.h")
set(UMBP_GRPC_SRCS "${UMBP_PROTO_GEN_DIR}/umbp.grpc.pb.cc")
set(UMBP_GRPC_HDRS "${UMBP_PROTO_GEN_DIR}/umbp.grpc.pb.h")

set(UMBP_PEER_PROTO_SRCS "${UMBP_PROTO_GEN_DIR}/umbp_peer.pb.cc")
set(UMBP_PEER_PROTO_HDRS "${UMBP_PROTO_GEN_DIR}/umbp_peer.pb.h")
set(UMBP_PEER_GRPC_SRCS "${UMBP_PROTO_GEN_DIR}/umbp_peer.grpc.pb.cc")
set(UMBP_PEER_GRPC_HDRS "${UMBP_PROTO_GEN_DIR}/umbp_peer.grpc.pb.h")

add_custom_command(
  OUTPUT ${UMBP_PROTO_SRCS} ${UMBP_PROTO_HDRS} ${UMBP_GRPC_SRCS}
         ${UMBP_GRPC_HDRS}
  COMMAND
    ${_PROTOBUF_PROTOC} --proto_path=${UMBP_PROTO_DIR}
    --cpp_out=${UMBP_PROTO_GEN_DIR} --grpc_out=${UMBP_PROTO_GEN_DIR}
    --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN} ${UMBP_PROTO_FILE}
  DEPENDS ${UMBP_PROTO_FILE}
  COMMENT "Generating UMBP protobuf/gRPC C++ sources"
  VERBATIM)

add_custom_command(
  OUTPUT ${UMBP_PEER_PROTO_SRCS} ${UMBP_PEER_PROTO_HDRS} ${UMBP_PEER_GRPC_SRCS}
         ${UMBP_PEER_GRPC_HDRS}
  COMMAND
    ${_PROTOBUF_PROTOC} --proto_path=${UMBP_PROTO_DIR}
    --cpp_out=${UMBP_PROTO_GEN_DIR} --grpc_out=${UMBP_PROTO_GEN_DIR}
    --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN} ${UMBP_PEER_PROTO_FILE}
  DEPENDS ${UMBP_PEER_PROTO_FILE}
  COMMENT "Generating UMBP Peer protobuf/gRPC C++ sources"
  VERBATIM)

# ---------- Static library: umbp_common ---------------------------------------
# Shared between PoolClient integration and standalone control-plane tools.

add_library(
  umbp_common STATIC
  ${UMBP_PROTO_SRCS}
  ${UMBP_GRPC_SRCS}
  ${UMBP_PEER_PROTO_SRCS}
  ${UMBP_PEER_GRPC_SRCS}
  distributed/master/global_block_index.cpp
  distributed/master/client_registry.cpp
  distributed/master/master_server.cpp
  distributed/master/master_client.cpp
  distributed/routing/router.cpp
  distributed/routing/route_get_strategy.cpp
  distributed/routing/route_put_strategy.cpp
  distributed/peer/peer_service.cpp
  distributed/pool_client.cpp)

target_include_directories(
  umbp_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${UMBP_PROTO_GEN_DIR}
                     ${_PROTOBUF_INCLUDE_DIR})

# mori_io pulls in Torch headers, which can shadow the protobuf headers used by
# protoc. Expose only the resolved google/ subtree ahead of Torch's includes.
set(_PROTOBUF_COMPAT_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf_compat")
if(NOT EXISTS "${_PROTOBUF_COMPAT_DIR}/google")
  file(MAKE_DIRECTORY "${_PROTOBUF_COMPAT_DIR}")
  file(CREATE_LINK "${_PROTOBUF_INCLUDE_DIR}/google"
       "${_PROTOBUF_COMPAT_DIR}/google" SYMBOLIC)
endif()
target_include_directories(umbp_common BEFORE PUBLIC "${_PROTOBUF_COMPAT_DIR}")

target_link_libraries(umbp_common PUBLIC mori_logging mori_io ${_PROTOBUF_LIBS}
                                         ${_GRPCPP_LIB})

target_compile_features(umbp_common PUBLIC cxx_std_17)

# Link umbp_core → umbp_common so UMBPClient can construct PoolClient and see
# the generated protobuf/gRPC headers during compilation.
target_link_libraries(umbp_core PUBLIC umbp_common)

# ---------- Master executable -------------------------------------------------

add_executable(umbp_master distributed/bin/master_main.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(umbp_master PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(umbp_master PRIVATE umbp_common ${_PROTOBUF_LIBS}
                                          ${_GRPCPP_LIB})

# ---------- Client executable -------------------------------------------------

add_executable(umbp_client distributed/bin/client_main.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(umbp_client PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(umbp_client PRIVATE umbp_common ${_PROTOBUF_LIBS}
                                          ${_GRPCPP_LIB})
