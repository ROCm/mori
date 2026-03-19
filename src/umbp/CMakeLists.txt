# Core static library
add_library(
  umbp_core STATIC
  block_index/local_block_index.cpp
  storage/io/storage_io_driver.cpp
  storage/io/posix_storage_io_driver.cpp
  storage/io/io_uring_storage_io_driver.cpp
  storage/copy_pipeline.cpp
  storage/segment/segment_format.cpp
  storage/segment/segment_index.cpp
  storage/segment/segment_scanner.cpp
  storage/segment/segment_writer.cpp
  storage/tier_backend.cpp
  storage/dram_tier.cpp
  storage/ssd_tier.cpp
  storage/local_storage_manager.cpp
  umbp_client.cpp)
target_include_directories(umbp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(umbp_core PRIVATE pthread)

# Python bindings are registered in src/pybind/pybind_umbp.cpp (compiled into
# libmori_pybinds when BUILD_UMBP=ON)

# =============================================================================
# UMBP Master Control Plane — CMake Build
# =============================================================================

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)

# ---------- Dependency resolution ---------------------------------------------

if(TARGET gRPC::grpc++)
  # Preferred path: use official gRPC CMake target if available.
  set(_GRPCPP_LIB gRPC::grpc++)
elseif(TARGET grpc++)
  # Some environments export un-namespaced targets.
  set(_GRPCPP_LIB grpc++)
else()
  # Fallback path: resolve local install by headers/libs and create an imported
  # target so downstream link code stays identical.
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

# Resolve protobuf deterministically from one local installation. Always use
# full runtime (libprotobuf), never lite.
find_library(_PROTOBUF_LIB NAMES protobuf libprotobuf REQUIRED)
get_filename_component(_PROTOBUF_LIB_DIR "${_PROTOBUF_LIB}" DIRECTORY)
get_filename_component(_PROTOBUF_PREFIX "${_PROTOBUF_LIB_DIR}" DIRECTORY)
find_path(_PROTOBUF_INCLUDE_DIR google/protobuf/message.h
          HINTS "${_PROTOBUF_PREFIX}/include" REQUIRED)
set(_PROTOBUF_LIBS "${_PROTOBUF_LIB}")
message(STATUS "UMBP protobuf lib: ${_PROTOBUF_LIBS}")
message(STATUS "UMBP protobuf include: ${_PROTOBUF_INCLUDE_DIR}")

# Keep protoc from the same install when possible to avoid
# generated-code/runtime ABI mismatches.
set(_PROTOBUF_PROTOC "${Protobuf_PROTOC_EXECUTABLE}")
if(NOT _PROTOBUF_PROTOC)
  set(_PROTOBUF_PROTOC "${PROTOBUF_PROTOC_EXECUTABLE}")
endif()
if(NOT _PROTOBUF_PROTOC)
  find_program(_PROTOBUF_PROTOC protoc HINTS "${_PROTOBUF_PREFIX}/bin" REQUIRED)
endif()
message(STATUS "UMBP protoc: ${_PROTOBUF_PROTOC}")

if(TARGET gRPC::grpc_cpp_plugin)
  # Use target-provided plugin path when gRPC package exports it.
  set(_GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
elseif(TARGET grpc_cpp_plugin)
  set(_GRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)
else()
  # Fallback to PATH lookup for distro-provided plugin binaries.
  find_program(_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
endif()

# Guard against mixed protobuf installations.
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

# ---------- Proto compilation -------------------------------------------------

set(UMBP_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
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
# Shared between master and client. Contains proto-generated code, types, and
# client_registry.

add_library(
  umbp_common STATIC
  ${UMBP_PROTO_SRCS}
  ${UMBP_GRPC_SRCS}
  ${UMBP_PEER_PROTO_SRCS}
  ${UMBP_PEER_GRPC_SRCS}
  src/types.cpp
  src/block_index.cpp
  src/client_registry.cpp
  src/route_get_strategy.cpp
  src/route_put_strategy.cpp
  src/router.cpp
  src/peer_service.cpp
  src/pool_client.cpp
  src/master_client.cpp
  src/master_server.cpp)

target_include_directories(
  umbp_common
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include # umbp/types.h,
                                             # umbp/client_registry.h, etc.
         ${UMBP_PROTO_GEN_DIR} # umbp.pb.h, umbp.grpc.pb.h
         ${_PROTOBUF_INCLUDE_DIR})

# mori_io is needed for IOEngine RDMA data plane in PoolClient. However, mori_io
# -> mori_application -> Torch propagates Torch's bundled protobuf headers which
# conflict with the system protobuf used by protoc. Work around: create a
# high-priority include directory that resolves to the system protobuf,
# overriding Torch's version.
set(_PROTOBUF_COMPAT_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf_compat")
if(NOT EXISTS "${_PROTOBUF_COMPAT_DIR}/google")
  file(MAKE_DIRECTORY "${_PROTOBUF_COMPAT_DIR}")
  file(CREATE_LINK "/usr/include/google" "${_PROTOBUF_COMPAT_DIR}/google"
       SYMBOLIC)
endif()
target_include_directories(umbp_common BEFORE PUBLIC "${_PROTOBUF_COMPAT_DIR}")

target_link_libraries(umbp_common PUBLIC mori_logging mori_io)

target_compile_features(umbp_common PUBLIC cxx_std_17)

# ---------- Master executable -------------------------------------------------

add_executable(umbp_master src/master_main.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Prevent lld/as-needed from discarding protobuf when symbols come from static
  # archives (libumbp_common.a -> generated *.pb.o).
  target_link_options(umbp_master PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(umbp_master PRIVATE umbp_common ${_PROTOBUF_LIBS}
                                          ${_GRPCPP_LIB})

# ---------- Client executable -------------------------------------------------

add_executable(umbp_client src/client_main.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Same linker retention rule as umbp_master.
  target_link_options(umbp_client PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(umbp_client PRIVATE umbp_common ${_PROTOBUF_LIBS}
                                          ${_GRPCPP_LIB})

# ---------- PoolClient demo executable ----------------------------------------

add_executable(umbp_pool_client src/pool_client_main.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(umbp_pool_client PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(umbp_pool_client PRIVATE umbp_common ${_PROTOBUF_LIBS}
                                               ${_GRPCPP_LIB})
