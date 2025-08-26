add_library(mori_io SHARED engine.cpp rdma/protocol.cpp
                           rdma/backend_impl_v1.cpp)

target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
  mori_io
  mori_application
  ibverbs
  mlx5
  hip::host
  hip::device
  msgpack-c-interface)
