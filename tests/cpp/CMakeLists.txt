add_executable(test_transport_tcp application/test_transport_tcp.cpp)

target_include_directories(test_transport_tcp
                           PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_transport_tcp PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_transport_tcp mori_application)

add_executable(test_transport_ibverbs application/test_transport_ibverbs.cpp)

target_include_directories(test_transport_ibverbs
                           PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_transport_ibverbs PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_transport_ibverbs mori_application)

if(BUILD_IO)
  add_executable(test_engine io/test_engine.cpp)

  target_include_directories(test_engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_engine PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_engine mori_application mori_io)

  add_executable(test_topology application/test_topology.cpp)

  target_include_directories(test_topology PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_topology PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_topology mori_application mori_io)
endif()

if(BUILD_OPS_DEVICE)
  find_package(MPI REQUIRED)
  add_executable(test_aot_launch ops/test_aot_launch.cpp)
  target_include_directories(test_aot_launch PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_aot_launch PUBLIC ${CMAKE_SOURCE_DIR})
  target_include_directories(test_aot_launch PUBLIC ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(test_aot_launch mori_ops mori_shmem mori_application hip::host ${MPI_CXX_LIBRARIES})
endif()
