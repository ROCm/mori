set(MORI_IO_SOURCES
    engine.cpp
    rdma/protocol.cpp
    rdma/backend_impl.cpp
    rdma/executor.cpp
    rdma/common.cpp
    rdma/ledger.cpp
    xgmi/backend_impl.cpp
    xgmi/hip_resource_pool.cpp
    fabric/backend_impl.cpp
    fabric/vpod_topology.cpp)

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
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
