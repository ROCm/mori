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

add_executable(test_umbp_page_bitmap_allocator test_page_bitmap_allocator.cpp)
target_link_libraries(test_umbp_page_bitmap_allocator PRIVATE umbp_common
                                                              gtest_main)
add_test(NAME umbp_page_bitmap_allocator
         COMMAND test_umbp_page_bitmap_allocator)

add_executable(test_umbp_finalize_idempotency test_finalize_idempotency.cpp)
target_link_libraries(test_umbp_finalize_idempotency PRIVATE umbp_common
                                                             gtest_main)
add_test(NAME umbp_finalize_idempotency COMMAND test_umbp_finalize_idempotency)

add_executable(test_umbp_eviction test_eviction.cpp)
target_link_libraries(test_umbp_eviction PRIVATE umbp_common gtest_main)
add_test(NAME umbp_eviction COMMAND test_umbp_eviction)

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
  test_umbp_peer_service PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB}
                                 ${_TEST_GRPCPP_LIB} gtest_main)
add_test(NAME umbp_peer_service COMMAND test_umbp_peer_service)

add_executable(test_umbp_cross_node_smoke test_cross_node_smoke.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_cross_node_smoke PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_cross_node_smoke
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)
add_test(NAME umbp_cross_node_smoke COMMAND test_umbp_cross_node_smoke)

add_executable(test_umbp_pool_client_batch_abort
               test_pool_client_batch_abort.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_pool_client_batch_abort PRIVATE
                      -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_pool_client_batch_abort
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)
add_test(NAME umbp_pool_client_batch_abort
         COMMAND test_umbp_pool_client_batch_abort)

add_executable(test_umbp_pool_client_batch_put test_pool_client_batch_put.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_pool_client_batch_put PRIVATE
                      -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_pool_client_batch_put
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)
add_test(NAME umbp_pool_client_batch_put
         COMMAND test_umbp_pool_client_batch_put)
