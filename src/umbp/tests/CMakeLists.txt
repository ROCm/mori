# ---------------------------------------------------------------------------
# UMBP unit tests — requires GTest and umbp_common
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)

# Prevent GoogleTest from overriding compiler/linker options when built as a
# subproject.
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ---------------------------------------------------------------------------
# test_external_kv_block_index
# ---------------------------------------------------------------------------
add_executable(test_external_kv_block_index test_external_kv_block_index.cpp)

target_link_libraries(test_external_kv_block_index PRIVATE umbp_common
                                                           GTest::gtest_main)

target_compile_features(test_external_kv_block_index PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(test_external_kv_block_index)

# ---------------------------------------------------------------------------
# test_client_registry_external_kv
# ---------------------------------------------------------------------------
add_executable(test_client_registry_external_kv
               test_client_registry_external_kv.cpp)

target_link_libraries(test_client_registry_external_kv
                      PRIVATE umbp_common GTest::gtest_main)

target_compile_features(test_client_registry_external_kv PRIVATE cxx_std_17)

gtest_discover_tests(test_client_registry_external_kv)

# ---------------------------------------------------------------------------
# test_peer_dram_allocator
# ---------------------------------------------------------------------------
add_executable(test_peer_dram_allocator test_peer_dram_allocator.cpp)

target_link_libraries(test_peer_dram_allocator PRIVATE umbp_common
                                                       GTest::gtest_main)

target_compile_features(test_peer_dram_allocator PRIVATE cxx_std_17)

gtest_discover_tests(test_peer_dram_allocator)

# ---------------------------------------------------------------------------
# test_global_block_index_events
# ---------------------------------------------------------------------------
add_executable(test_global_block_index_events
               test_global_block_index_events.cpp)

target_link_libraries(test_global_block_index_events PRIVATE umbp_common
                                                             GTest::gtest_main)

target_compile_features(test_global_block_index_events PRIVATE cxx_std_17)

gtest_discover_tests(test_global_block_index_events)

# ---------------------------------------------------------------------------
# test_client_metric_dispatcher
# ---------------------------------------------------------------------------
add_executable(test_client_metric_dispatcher test_client_metric_dispatcher.cpp)

target_link_libraries(test_client_metric_dispatcher PRIVATE umbp_common
                                                            GTest::gtest_main)

target_compile_features(test_client_metric_dispatcher PRIVATE cxx_std_17)

gtest_discover_tests(test_client_metric_dispatcher)

# ---------------------------------------------------------------------------
# test_client_counter_rate_view
# ---------------------------------------------------------------------------
add_executable(test_client_counter_rate_view test_client_counter_rate_view.cpp)

target_link_libraries(test_client_counter_rate_view PRIVATE umbp_common
                                                            GTest::gtest_main)

target_compile_features(test_client_counter_rate_view PRIVATE cxx_std_17)

gtest_discover_tests(test_client_counter_rate_view)

# ---------------------------------------------------------------------------
# test_master_server_transfer_rates
# ---------------------------------------------------------------------------
add_executable(test_master_server_transfer_rates
               test_master_server_transfer_rates.cpp)

target_link_libraries(test_master_server_transfer_rates
                      PRIVATE umbp_common GTest::gtest_main)

target_compile_features(test_master_server_transfer_rates PRIVATE cxx_std_17)

gtest_discover_tests(test_master_server_transfer_rates)

# ---------------------------------------------------------------------------
# test_router_dedup — master-side BatchRoutePut dedup via GlobalBlockIndex
# ---------------------------------------------------------------------------
add_executable(test_router_dedup test_router_dedup.cpp)

target_link_libraries(test_router_dedup PRIVATE umbp_common GTest::gtest_main)

target_compile_features(test_router_dedup PRIVATE cxx_std_17)

gtest_discover_tests(test_router_dedup)
