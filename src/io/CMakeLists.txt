add_library(
  mori_io SHARED
  engine.cpp
  rdma/protocol.cpp
  rdma/backend_impl.cpp
  rdma/executor.cpp
  rdma/common.cpp
  xgmi/backend_impl.cpp
  xgmi/hip_resource_pool.cpp)

target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(
  mori_io PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/msgpack-c/include)
target_link_libraries(
  mori_io
  mori_application
  ibverbs
  mlx5
  hip::host
  hip::device
  mori_logging)
target_compile_definitions(mori_io PUBLIC MSGPACK_NO_BOOST)
