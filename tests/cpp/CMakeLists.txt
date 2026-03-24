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

add_executable(test_engine io/test_engine.cpp)

target_include_directories(test_engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_engine PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_engine mori_application mori_io)

add_executable(test_topology application/test_topology.cpp)

target_include_directories(test_topology PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_topology PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_topology mori_application mori_io)

add_executable(test_discrete_buffer io/test_discrete_buffer.cpp)
set_source_files_properties(io/test_discrete_buffer.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_discrete_buffer
                           PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_discrete_buffer PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_discrete_buffer hip::host hip::device)
if(DEFINED GPU_TARGETS)
  set_target_properties(test_discrete_buffer PROPERTIES HIP_ARCHITECTURES
                                                        "${GPU_TARGETS}")
endif()

if(BUILD_UMBP)
  add_subdirectory(umbp)
endif()
