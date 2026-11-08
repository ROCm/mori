set(MORI_IO_SOURCES
    engine.cpp
    peer_failure_tracker.cpp
    rdma/protocol.cpp
    rdma/backend_impl.cpp
    rdma/async_event_monitor.cpp
    rdma/executor.cpp
    rdma/common.cpp
    rdma/ledger.cpp
    sim/backend_impl.cpp
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
# With BUILD_RAS=OFF, include/mori/ras/ras.hpp compiles every RAS call in engine.cpp away.
if(BUILD_RAS)
  target_link_libraries(mori_io mori_ras)
endif()
target_compile_definitions(mori_io PUBLIC MSGPACK_NO_BOOST)
if(ENABLE_IO_FAULT_INJECTION)
  target_sources(mori_io PRIVATE rdma/fault_injector.cpp)
  set_source_files_properties(rdma/fault_injector.cpp PROPERTIES LANGUAGE CXX)
  target_compile_definitions(mori_io PUBLIC MORI_IO_FAULT_INJECTION)
endif()
set_target_properties(
  mori_io
  PROPERTIES BUILD_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             INSTALL_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)

# ── libfabric (OFI backend) ─────────────────────────────────────────────────
# Requested via MORI_USE_LIBFABRIC, but never a hard build/runtime dependency.
# libfabric is loaded at runtime via dlopen (see ofi/ofi_shim.cpp), never linked,
# so libmori_io.so carries no DT_NEEDED on libfabric.so. pkg-config is used only
# to locate the headers (include path). If the headers are missing we warn and
# build without the OFI backend. The MORI_USE_LIBFABRIC compile definition (which
# guards the OFI code paths) is only set when the headers are found, so intent
# and availability stay consistent.
if(MORI_USE_LIBFABRIC)
  find_package(PkgConfig)
  if(PkgConfig_FOUND)
    pkg_check_modules(LIBFABRIC libfabric)
  endif()
  if(LIBFABRIC_FOUND)
    message(STATUS "libfabric include: ${LIBFABRIC_INCLUDE_DIRS}")

    # libfabric is dlopen'ed at runtime (see ofi_shim.cpp), not linked. The shim
    # is an object library with hidden visibility so mori_io embeds its own
    # private copy of the forwarded fi_* symbols and never interposes a real
    # libfabric that may also be present in the process.
    add_library(mori_ofi_shim OBJECT ofi/ofi_shim.cpp)
    set_source_files_properties(ofi/ofi_shim.cpp PROPERTIES LANGUAGE CXX)
    target_include_directories(mori_ofi_shim PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                     ${LIBFABRIC_INCLUDE_DIRS})
    target_link_libraries(mori_ofi_shim PRIVATE mori_logging dl)
    target_compile_options(mori_ofi_shim PRIVATE -fvisibility=hidden
                                                 -fvisibility-inlines-hidden)
    set_target_properties(mori_ofi_shim PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_sources(mori_io PRIVATE ofi/backend_impl.cpp)
    target_include_directories(mori_io PRIVATE ${LIBFABRIC_INCLUDE_DIRS})
    target_link_libraries(mori_io mori_ofi_shim)
    target_compile_definitions(mori_io PUBLIC MORI_USE_LIBFABRIC)
  else()
    message(WARNING
      "MORI_USE_LIBFABRIC=ON but libfabric headers were not found (pkg-config "
      "module 'libfabric' missing); building WITHOUT the OFI backend. Install "
      "libfabric-dev, or pass -DMORI_USE_LIBFABRIC=OFF to silence this.")
  endif()
endif()
