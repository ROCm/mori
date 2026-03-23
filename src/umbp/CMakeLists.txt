# ---------------------------------------------------------------------------
# SPDK auto-detection: try pkg-config, enable if found.
# Set SPDK_PKG_CONFIG_PATH to override search path for non-standard installs.
# Set USE_SPDK=OFF to force-disable even if SPDK is installed.
# ---------------------------------------------------------------------------
set(SPDK_PKG_CONFIG_PATH "" CACHE STRING "Custom pkg-config path for SPDK")
option(USE_SPDK "Enable SPDK support (auto-detected by default)" ON)

set(HAVE_SPDK FALSE)
if(USE_SPDK)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    if(SPDK_PKG_CONFIG_PATH)
      set(ENV{PKG_CONFIG_PATH} "${SPDK_PKG_CONFIG_PATH}:$ENV{PKG_CONFIG_PATH}")
    endif()

    execute_process(
      COMMAND pkg-config --cflags spdk_event spdk_event_bdev spdk_env_dpdk
      OUTPUT_VARIABLE SPDK_CFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE SPDK_CFLAGS_RC
      ERROR_QUIET)
    execute_process(
      COMMAND pkg-config --libs --static spdk_event spdk_event_bdev spdk_env_dpdk
      OUTPUT_VARIABLE SPDK_LDFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE SPDK_LDFLAGS_RC
      ERROR_QUIET)

    if(SPDK_CFLAGS_RC EQUAL 0 AND SPDK_LDFLAGS_RC EQUAL 0)
      set(HAVE_SPDK TRUE)
    endif()
  endif()

  if(HAVE_SPDK)
    message(STATUS "[UMBP] SPDK detected via pkg-config — SPDK support ENABLED")
    message(STATUS "[UMBP] SPDK CFLAGS: ${SPDK_CFLAGS}")
    message(STATUS "[UMBP] SPDK LDFLAGS: ${SPDK_LDFLAGS}")
  else()
    message(STATUS "[UMBP] SPDK not found via pkg-config — SPDK support DISABLED (POSIX only)")
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
  local/storage/local_storage_manager.cpp
  local/umbp_client.cpp
  allocator/offset_allocator.cpp)

# SPDK Proxy client (shared memory IPC, no SPDK dependency — Linux only)
if(UNIX)
  list(APPEND UMBP_CORE_SRCS
    proxy/spdk_proxy_shm.cpp
    local/storage/spdk_proxy_tier.cpp)
endif()

if(HAVE_SPDK)
  list(APPEND UMBP_CORE_SRCS
    local/storage/spdk/spdk_env.cpp
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
  target_link_libraries(umbp_core PUBLIC
    -Wl,--whole-archive ${SPDK_LDFLAGS_LIST} -Wl,--no-whole-archive
    -lisal -lisal_crypto -lssl -lcrypto -lfuse3 -lnuma -ldl -lrt -luuid -lm -laio -lpthread -lelf)

  # SPDK test executables
  add_executable(test_spdk_env ${CMAKE_SOURCE_DIR}/tests/cpp/umbp/local/test_spdk_env.cpp)
  target_link_libraries(test_spdk_env PRIVATE umbp_core)

  add_executable(test_spdk_ssd_tier ${CMAKE_SOURCE_DIR}/tests/cpp/umbp/local/test_spdk_ssd_tier.cpp)
  target_link_libraries(test_spdk_ssd_tier PRIVATE umbp_core)

  # SPDK benchmark
  add_executable(spdk_bench benchmarks/spdk_bench.cpp)
  target_link_libraries(spdk_bench PRIVATE umbp_core)

  # SPDK Proxy daemon (owns SPDK, serves ranks via SHM IPC)
  add_executable(spdk_proxy proxy/spdk_proxy_daemon.cpp)
  target_link_libraries(spdk_proxy PRIVATE umbp_core)
endif()

# SPDK Proxy benchmark (no SPDK dependency, only SHM — Linux only)
if(UNIX)
  add_executable(proxy_bench benchmarks/proxy_bench.cpp)
  target_link_libraries(proxy_bench PRIVATE umbp_core)

  # E2E benchmark using UMBPClient (auto Leader/Follower fork)
  add_executable(umbp_bench benchmarks/umbp_bench.cpp)
  target_link_libraries(umbp_bench PRIVATE umbp_core)

  # E2E integration test (one command, all backends)
  add_executable(test_umbp_e2e ${CMAKE_SOURCE_DIR}/tests/cpp/umbp/local/test_umbp_e2e.cpp)
  target_link_libraries(test_umbp_e2e PRIVATE umbp_core)
endif()

# Python bindings are registered in src/pybind/pybind_umbp.cpp (compiled into
# libmori_pybinds when BUILD_UMBP=ON)

# =============================================================================
# UMBP Master Control Plane — CMake Build (optional: requires gRPC + Protobuf)
# =============================================================================

find_package(Protobuf QUIET)
find_package(gRPC CONFIG QUIET)

set(HAVE_GRPC FALSE)

if(Protobuf_FOUND)
  # ---------- Dependency resolution -------------------------------------------
  if(TARGET gRPC::grpc++)
    set(_GRPCPP_LIB gRPC::grpc++)
    set(HAVE_GRPC TRUE)
  elseif(TARGET grpc++)
    set(_GRPCPP_LIB grpc++)
    set(HAVE_GRPC TRUE)
  else()
    find_path(GRPC_INCLUDE_DIR grpcpp/grpcpp.h QUIET)
    find_library(GRPCPP_LIBRARY NAMES grpc++ QUIET)
    find_library(GRPC_LIBRARY NAMES grpc QUIET)
    find_library(GPR_LIBRARY NAMES gpr QUIET)

    if(GRPC_INCLUDE_DIR AND GRPCPP_LIBRARY AND GRPC_LIBRARY AND GPR_LIBRARY)
      add_library(gRPC::grpc++ UNKNOWN IMPORTED)
      set_target_properties(
        gRPC::grpc++
        PROPERTIES IMPORTED_LOCATION "${GRPCPP_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${GRPC_INCLUDE_DIR}"
                   INTERFACE_LINK_LIBRARIES "${GRPC_LIBRARY};${GPR_LIBRARY}")
      set(_GRPCPP_LIB gRPC::grpc++)
      set(HAVE_GRPC TRUE)
    endif()
  endif()
endif()

if(HAVE_GRPC)
  message(STATUS "[UMBP] gRPC + Protobuf found — distributed control plane ENABLED")

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

  if(DEFINED Protobuf_VERSION)
    execute_process(
      COMMAND ${_PROTOBUF_PROTOC} --version
      OUTPUT_VARIABLE _PROTOC_VERSION_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _PROTOC_MM "${_PROTOC_VERSION_OUTPUT}")
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _PROTOBUF_MM "${Protobuf_VERSION}")
    if(_PROTOC_MM
       AND _PROTOBUF_MM
       AND NOT _PROTOC_MM STREQUAL _PROTOBUF_MM)
      message(
        FATAL_ERROR
          "protoc (${_PROTOC_VERSION_OUTPUT}) and libprotobuf (${Protobuf_VERSION}) mismatch."
      )
    endif()
  endif()

  # ---------- Proto compilation -----------------------------------------------

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

  # ---------- Static library: umbp_common -------------------------------------

  add_library(
    umbp_common STATIC
    ${UMBP_PROTO_SRCS}
    ${UMBP_GRPC_SRCS}
    ${UMBP_PEER_PROTO_SRCS}
    ${UMBP_PEER_GRPC_SRCS}
    distributed/common/types.cpp
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
    umbp_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
           ${UMBP_PROTO_GEN_DIR}
           ${_PROTOBUF_INCLUDE_DIR})

  set(_PROTOBUF_COMPAT_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf_compat")
  if(NOT EXISTS "${_PROTOBUF_COMPAT_DIR}/google")
    file(MAKE_DIRECTORY "${_PROTOBUF_COMPAT_DIR}")
    file(CREATE_LINK "/usr/include/google" "${_PROTOBUF_COMPAT_DIR}/google"
         SYMBOLIC)
  endif()
  target_include_directories(umbp_common BEFORE PUBLIC "${_PROTOBUF_COMPAT_DIR}")

  target_link_libraries(umbp_common PUBLIC mori_logging mori_io)

  target_compile_features(umbp_common PUBLIC cxx_std_17)

  # ---------- Master executable -----------------------------------------------

  add_executable(umbp_master distributed/bin/master_main.cpp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(umbp_master PRIVATE -Wl,--no-as-needed)
  endif()
  target_link_libraries(umbp_master PRIVATE umbp_common ${_PROTOBUF_LIBS}
                                            ${_GRPCPP_LIB})

  # ---------- Client executable -----------------------------------------------

  add_executable(umbp_client distributed/bin/client_main.cpp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(umbp_client PRIVATE -Wl,--no-as-needed)
  endif()
  target_link_libraries(umbp_client PRIVATE umbp_common ${_PROTOBUF_LIBS}
                                            ${_GRPCPP_LIB})

  # ---------- PoolClient demo executable --------------------------------------

  add_executable(umbp_pool_client distributed/bin/pool_client_main.cpp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(umbp_pool_client PRIVATE -Wl,--no-as-needed)
  endif()
  target_link_libraries(umbp_pool_client PRIVATE umbp_common ${_PROTOBUF_LIBS}
                                                 ${_GRPCPP_LIB})

else()
  message(STATUS "[UMBP] gRPC/Protobuf not found — distributed control plane DISABLED (local-only build)")
endif()
