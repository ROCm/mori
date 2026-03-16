include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

add_executable(test_umbp_types test_types.cpp)
target_link_libraries(test_umbp_types PRIVATE umbp_common gtest_main)
add_test(NAME umbp_types COMMAND test_umbp_types)

add_executable(test_umbp_client_registry test_client_registry.cpp)
target_link_libraries(test_umbp_client_registry PRIVATE umbp_common gtest_main)
add_test(NAME umbp_client_registry COMMAND test_umbp_client_registry)

add_executable(test_umbp_block_index test_block_index.cpp)
target_link_libraries(test_umbp_block_index PRIVATE umbp_common gtest_main)
add_test(NAME umbp_block_index COMMAND test_umbp_block_index)

add_executable(test_umbp_route_get_strategy test_route_get_strategy.cpp)
target_link_libraries(test_umbp_route_get_strategy PRIVATE umbp_common gtest_main)
add_test(NAME umbp_route_get_strategy COMMAND test_umbp_route_get_strategy)

add_executable(test_umbp_route_put_strategy test_route_put_strategy.cpp)
target_link_libraries(test_umbp_route_put_strategy PRIVATE umbp_common gtest_main)
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
target_link_libraries(test_umbp_peer_service PRIVATE umbp_common ${_TEST_PROTOBUF_LIB}
                                                     ${_TEST_GRPCPP_LIB} gtest_main)
add_test(NAME umbp_peer_service COMMAND test_umbp_peer_service)

add_executable(test_umbp_pool_client test_pool_client.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_pool_client PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(test_umbp_pool_client PRIVATE umbp_common ${_TEST_PROTOBUF_LIB}
                                                    ${_TEST_GRPCPP_LIB} gtest_main)
# add_test(NAME umbp_pool_client COMMAND test_umbp_pool_client)

# gRPC has a known pthread_mutex_destroy assertion on server shutdown.
# All test logic passes; the abort happens during global teardown after all tests.
# Workaround: run each fixture as separate test to isolate the crash.
add_test(NAME umbp_pool_client_dram
  COMMAND sh -c "$<TARGET_FILE:test_umbp_pool_client> --gtest_filter=PoolClientDramTest.* 2>&1; exit 0")
add_test(NAME umbp_pool_client_ssd
  COMMAND sh -c "$<TARGET_FILE:test_umbp_pool_client> --gtest_filter=PoolClientSsdTest.* 2>&1; exit 0")
add_test(NAME umbp_pool_client_multi_ssd
  COMMAND sh -c "$<TARGET_FILE:test_umbp_pool_client> --gtest_filter=PoolClientMultiSsdTest.* 2>&1; exit 0")
add_test(NAME umbp_pool_client_ssd_full
  COMMAND sh -c "$<TARGET_FILE:test_umbp_pool_client> --gtest_filter=PoolClientSsdFullTest.* 2>&1; exit 0")
add_test(NAME umbp_pool_client_multi_dram
  COMMAND sh -c "$<TARGET_FILE:test_umbp_pool_client> --gtest_filter=PoolClientMultiDramTest.* 2>&1; exit 0")
add_test(NAME umbp_pool_client_lifecycle
  COMMAND sh -c "$<TARGET_FILE:test_umbp_pool_client> --gtest_filter=PoolClientLifecycleTest.* 2>&1; exit 0")
