add_executable(test_umbp_types test_types.cpp)
target_link_libraries(test_umbp_types PRIVATE umbp_common gtest_main)
add_test(NAME umbp_types COMMAND test_umbp_types)

add_executable(test_umbp_client_registry test_client_registry.cpp)
target_link_libraries(test_umbp_client_registry PRIVATE umbp_common gtest_main)
add_test(NAME umbp_client_registry COMMAND test_umbp_client_registry)

add_executable(test_umbp_global_block_index test_global_block_index.cpp)
target_link_libraries(test_umbp_global_block_index PRIVATE umbp_common
                                                           gtest_main)
add_test(NAME umbp_global_block_index COMMAND test_umbp_global_block_index)

add_executable(test_umbp_route_get_strategy test_route_get_strategy.cpp)
target_link_libraries(test_umbp_route_get_strategy PRIVATE umbp_common
                                                           gtest_main)
add_test(NAME umbp_route_get_strategy COMMAND test_umbp_route_get_strategy)

add_executable(test_umbp_route_put_strategy test_route_put_strategy.cpp)
target_link_libraries(test_umbp_route_put_strategy PRIVATE umbp_common
                                                           gtest_main)
add_test(NAME umbp_route_put_strategy COMMAND test_umbp_route_put_strategy)

add_executable(test_umbp_router test_router.cpp)
target_link_libraries(test_umbp_router PRIVATE umbp_common gtest_main)
add_test(NAME umbp_router COMMAND test_umbp_router)

add_executable(test_umbp_pool_allocator test_pool_allocator.cpp)
target_link_libraries(test_umbp_pool_allocator PRIVATE umbp_common gtest_main)
add_test(NAME umbp_pool_allocator COMMAND test_umbp_pool_allocator)

find_library(_TEST_PROTOBUF_LIB NAMES protobuf libprotobuf REQUIRED)
if(TARGET gRPC::grpc++)
  set(_TEST_GRPCPP_LIB gRPC::grpc++)
elseif(TARGET grpc++)
  set(_TEST_GRPCPP_LIB grpc++)
else()
  find_library(_TEST_GRPCPP_LIB NAMES grpc++ REQUIRED)
endif()

add_executable(test_umbp_peer_service test_peer_service.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_peer_service PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_peer_service PRIVATE umbp_common ${_TEST_PROTOBUF_LIB}
                                 ${_TEST_GRPCPP_LIB} gtest_main)
add_test(NAME umbp_peer_service COMMAND test_umbp_peer_service)
