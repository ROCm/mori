set(MORI_IO_SOURCES
    engine.cpp
    rdma/protocol.cpp
    rdma/backend_impl.cpp
    rdma/async_event_monitor.cpp
    rdma/executor.cpp
    rdma/common.cpp
    rdma/ledger.cpp
    xgmi/backend_impl.cpp
    xgmi/hip_resource_pool.cpp
    fabric/backend_impl.cpp
    fabric/vpod_topology.cpp
    fabric/torch_allocator.cpp)

set_source_files_properties(${MORI_IO_SOURCES} PROPERTIES LANGUAGE CXX)
add_library(mori_io SHARED ${MORI_IO_SOURCES})

target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(
  mori_io PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/msgpack-c/include)
# libibverbs is loaded at runtime via dlopen; embed the shim (hidden visibility)
# so mori_io's own ibv_* references resolve without linking libibverbs.
target_link_libraries(mori_io mori_application hip::host mori_logging
                      mori_ibv_shim)
target_compile_definitions(mori_io PUBLIC MSGPACK_NO_BOOST)
set_target_properties(
  mori_io
  PROPERTIES BUILD_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             INSTALL_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)

# ── libfabric (OFI backend) ─────────────────────────────────────────────────
# Requested via MORI_USE_LIBFABRIC, but never a hard build dependency: if
# libfabric is missing we warn and build without the OFI backend. The
# MORI_USE_LIBFABRIC compile definition (which guards the OFI code paths) is
# only set when libfabric is actually found, so intent and availability stay
# consistent.
if(MORI_USE_LIBFABRIC)
  find_package(PkgConfig)
  if(PkgConfig_FOUND)
    pkg_check_modules(LIBFABRIC libfabric)
  endif()
  if(LIBFABRIC_FOUND)
    message(STATUS "libfabric include: ${LIBFABRIC_INCLUDE_DIRS}")
    message(STATUS "libfabric lib:     ${LIBFABRIC_LIBRARIES}")
    target_sources(mori_io PRIVATE ofi/backend_impl.cpp)
    target_include_directories(mori_io PRIVATE ${LIBFABRIC_INCLUDE_DIRS})
    target_link_libraries(mori_io ${LIBFABRIC_LIBRARIES})
    target_compile_definitions(mori_io PUBLIC MORI_USE_LIBFABRIC)
    target_link_directories(mori_io PUBLIC ${LIBFABRIC_LIBRARY_DIRS})
  else()
    message(WARNING
      "MORI_USE_LIBFABRIC=ON but libfabric was not found (pkg-config module "
      "'libfabric' missing); building WITHOUT the OFI backend. Install "
      "libfabric-dev, or pass -DMORI_USE_LIBFABRIC=OFF to silence this.")
  endif()
endif()
