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

if(ENABLE_XLA_FFI)
  add_executable(test_ffi_contract ffi/test_ffi_contract.cpp)
  target_include_directories(
    test_ffi_contract
    PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/)
  target_link_libraries(test_ffi_contract mori_xla_ffi_ops hip::host hip::device)
endif()
