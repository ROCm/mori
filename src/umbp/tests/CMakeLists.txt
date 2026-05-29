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

include(GoogleTest)

# ---------------------------------------------------------------------------
# test_external_kv_block_index
# ---------------------------------------------------------------------------
add_executable(test_external_kv_block_index test_external_kv_block_index.cpp)

target_link_libraries(test_external_kv_block_index PRIVATE umbp_common
                                                           GTest::gtest_main)

target_compile_features(test_external_kv_block_index PRIVATE cxx_std_17)

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
# test_external_kv_hit_index
# ---------------------------------------------------------------------------
add_executable(test_external_kv_hit_index test_external_kv_hit_index.cpp)

target_link_libraries(test_external_kv_hit_index PRIVATE umbp_common
                                                         GTest::gtest_main)

target_compile_features(test_external_kv_hit_index PRIVATE cxx_std_17)

gtest_discover_tests(test_external_kv_hit_index)

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
# test_router_dedup — master-side BatchRoutePut dedup via GlobalBlockIndex
# ---------------------------------------------------------------------------
add_executable(test_router_dedup test_router_dedup.cpp)

target_link_libraries(test_router_dedup PRIVATE umbp_common GTest::gtest_main)

target_compile_features(test_router_dedup PRIVATE cxx_std_17)

gtest_discover_tests(test_router_dedup)

# ---------------------------------------------------------------------------
# test_peer_ssd_manager — Phase 1 SSD tier ownership + owned-location source
# ---------------------------------------------------------------------------
add_executable(test_peer_ssd_manager test_peer_ssd_manager.cpp)

target_link_libraries(test_peer_ssd_manager PRIVATE umbp_common
                                                    GTest::gtest_main)

target_compile_features(test_peer_ssd_manager PRIVATE cxx_std_17)

gtest_discover_tests(test_peer_ssd_manager)

# ---------------------------------------------------------------------------
# test_ssd_copy_pipeline — Phase 2 DramCopyPin + async copy-on-commit pipeline
# ---------------------------------------------------------------------------
add_executable(test_ssd_copy_pipeline test_ssd_copy_pipeline.cpp)

target_link_libraries(test_ssd_copy_pipeline PRIVATE umbp_common
                                                     GTest::gtest_main)

target_compile_features(test_ssd_copy_pipeline PRIVATE cxx_std_17)

gtest_discover_tests(test_ssd_copy_pipeline)
