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
# test_master_metadata_store_interface — Phase 1 compile/instantiation gate for
# IMasterMetadataStore. Includes an isolated self-compile TU (proves the header
# is self-contained) and a GMock mock instantiation/signature-completeness test.
# ---------------------------------------------------------------------------
add_executable(
  test_master_metadata_store_interface test_master_metadata_store_interface.cpp
                                       master_metadata_store_self_compile.cpp)

target_include_directories(test_master_metadata_store_interface
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(test_master_metadata_store_interface
                      PRIVATE umbp_common GTest::gmock GTest::gtest_main)

target_compile_features(test_master_metadata_store_interface PRIVATE cxx_std_17)

gtest_discover_tests(test_master_metadata_store_interface)

# ---------------------------------------------------------------------------
# test_in_memory_master_metadata_store — Phase 2 behavioral suite for the
# InMemoryMasterMetadataStore implementation of IMasterMetadataStore (§6a).
# Written against IMasterMetadataStore& so it is reused for the Redis backend.
# ---------------------------------------------------------------------------
add_executable(test_in_memory_master_metadata_store
               test_in_memory_master_metadata_store.cpp)

target_link_libraries(test_in_memory_master_metadata_store
                      PRIVATE umbp_common GTest::gtest_main)

target_compile_features(test_in_memory_master_metadata_store PRIVATE cxx_std_17)

gtest_discover_tests(test_in_memory_master_metadata_store)

# ---------------------------------------------------------------------------
# test_external_kv_block_index
# ---------------------------------------------------------------------------
add_executable(test_external_kv_block_index test_external_kv_block_index.cpp)

target_link_libraries(test_external_kv_block_index PRIVATE umbp_common
                                                           GTest::gtest_main)

target_compile_features(test_external_kv_block_index PRIVATE cxx_std_17)

gtest_discover_tests(test_external_kv_block_index)

# ---------------------------------------------------------------------------
# test_client_registry — membership ledger: register/re-register, capacity
# round-trip, heartbeat status, and the silent-node reaper.
# ---------------------------------------------------------------------------
add_executable(test_client_registry test_client_registry.cpp)

target_link_libraries(test_client_registry PRIVATE umbp_common
                                                   GTest::gtest_main)

target_compile_features(test_client_registry PRIVATE cxx_std_17)

gtest_discover_tests(test_client_registry)

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
# test_router_dedup — master-side BatchRoutePut dedup via IMasterMetadataStore
# ---------------------------------------------------------------------------
add_executable(test_router_dedup test_router_dedup.cpp)

target_link_libraries(test_router_dedup PRIVATE umbp_common GTest::gtest_main)

target_compile_features(test_router_dedup PRIVATE cxx_std_17)

gtest_discover_tests(test_router_dedup)

# ---------------------------------------------------------------------------
# test_peer_ssd_manager — SSD tier ownership + owned-location source
# ---------------------------------------------------------------------------
add_executable(test_peer_ssd_manager test_peer_ssd_manager.cpp)

target_link_libraries(test_peer_ssd_manager PRIVATE umbp_common
                                                    GTest::gtest_main)

target_compile_features(test_peer_ssd_manager PRIVATE cxx_std_17)

gtest_discover_tests(test_peer_ssd_manager)

# ---------------------------------------------------------------------------
# test_peer_ssd_eviction — LRU + watermark eviction + in-flight guard
# ---------------------------------------------------------------------------
add_executable(test_peer_ssd_eviction test_peer_ssd_eviction.cpp)

target_link_libraries(test_peer_ssd_eviction PRIVATE umbp_common
                                                     GTest::gtest_main)

target_compile_features(test_peer_ssd_eviction PRIVATE cxx_std_17)

gtest_discover_tests(test_peer_ssd_eviction)

# ---------------------------------------------------------------------------
# test_ssd_copy_pipeline — DramCopyPin + async copy-on-commit pipeline
# ---------------------------------------------------------------------------
add_executable(test_ssd_copy_pipeline test_ssd_copy_pipeline.cpp)

target_link_libraries(test_ssd_copy_pipeline PRIVATE umbp_common
                                                     GTest::gtest_main)

target_compile_features(test_ssd_copy_pipeline PRIVATE cxx_std_17)

gtest_discover_tests(test_ssd_copy_pipeline)

# ---------------------------------------------------------------------------
# test_tier_priority_route_get — RouteGet tier-priority strategy
# ---------------------------------------------------------------------------
add_executable(test_tier_priority_route_get test_tier_priority_route_get.cpp)

target_link_libraries(test_tier_priority_route_get PRIVATE umbp_common
                                                           GTest::gtest_main)

target_compile_features(test_tier_priority_route_get PRIVATE cxx_std_17)

gtest_discover_tests(test_tier_priority_route_get)

# ---------------------------------------------------------------------------
# test_peer_ssd_read_rpc — SSD read RPC over a gRPC loopback (status
# distinctions: OK / NOT_FOUND / NO_SLOT / SIZE_TOO_LARGE + slot lifecycle).
# Needs the gRPC peer service + generated stubs (umbp_core) and protobuf/gRPC.
# ---------------------------------------------------------------------------
add_executable(test_peer_ssd_read_rpc test_peer_ssd_read_rpc.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_peer_ssd_read_rpc PRIVATE -Wl,--no-as-needed)
endif()

target_link_libraries(
  test_peer_ssd_read_rpc PRIVATE umbp_core umbp_common ${_PROTOBUF_LIB}
                                 ${_GRPCPP_LIB} GTest::gtest_main)

target_compile_features(test_peer_ssd_read_rpc PRIVATE cxx_std_17)

gtest_discover_tests(test_peer_ssd_read_rpc)

# ---------------------------------------------------------------------------
# test_batch_resolve_codec — struct-of-arrays BatchResolveKeysResponse wire
# codec (batch_resolve_codec.h): encode/decode round-trip, page flattening, desc
# dedup / omit_descs, malformed-response rejection, and the payload-size win.
# Needs the generated proto (carried by umbp_common) + protobuf.
# ---------------------------------------------------------------------------
add_executable(test_batch_resolve_codec test_batch_resolve_codec.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_batch_resolve_codec PRIVATE -Wl,--no-as-needed)
endif()

target_link_libraries(test_batch_resolve_codec
                      PRIVATE umbp_common ${_PROTOBUF_LIB} GTest::gtest_main)

target_compile_features(test_batch_resolve_codec PRIVATE cxx_std_17)

gtest_discover_tests(test_batch_resolve_codec)

# ---------------------------------------------------------------------------
# test_ssd_read_lease_gating — pure reader-side lease gating decision logic
# (ssd_read_lease.h).  Header-only under test (no gRPC / RDMA deps).
# ---------------------------------------------------------------------------
add_executable(test_ssd_read_lease_gating test_ssd_read_lease_gating.cpp)

target_link_libraries(test_ssd_read_lease_gating PRIVATE umbp_common
                                                         GTest::gtest_main)

target_compile_features(test_ssd_read_lease_gating PRIVATE cxx_std_17)

gtest_discover_tests(test_ssd_read_lease_gating)

# ---------------------------------------------------------------------------
# test_ssd_reliability — cross-component reliability: owned-source DRAM+SSD
# merge, SSD evict -> REMOVE -> master index convergence, tier-priority over the
# real index, crash-restart discard, and observability counters.
# ---------------------------------------------------------------------------
add_executable(test_ssd_reliability test_ssd_reliability.cpp)

target_link_libraries(test_ssd_reliability PRIVATE umbp_common
                                                   GTest::gtest_main)

target_compile_features(test_ssd_reliability PRIVATE cxx_std_17)

gtest_discover_tests(test_ssd_reliability)
