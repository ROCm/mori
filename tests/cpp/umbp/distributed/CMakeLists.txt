# --------------------------------------------------------------------------
# Tests in this directory compile against the current MasterClient /
# MasterServer / PoolClient / PeerDramAllocator surface (Route* / Batch* /
# event-driven index). They are split into three groups:
#
# * unit        - pure-logic tests (no gRPC / RDMA); always registered as CTest
#   cases.
# * integration - stand up in-process master/peer servers and exercise the gRPC
#   + RDMA data path; registered as CTest cases under the "integration" label so
#   environments without the required transport can skip them via `ctest -LE
#   integration`.
# * bench       - manual micro-benchmarks; built but NOT registered with
#   add_test (never run under a plain `ctest`).
# --------------------------------------------------------------------------

find_library(_TEST_PROTOBUF_LIB NAMES protobuf libprotobuf REQUIRED)
if(TARGET gRPC::grpc++)
  set(_TEST_GRPCPP_LIB gRPC::grpc++)
elseif(TARGET grpc++)
  set(_TEST_GRPCPP_LIB grpc++)
else()
  find_library(_TEST_GRPCPP_LIB NAMES grpc++ REQUIRED)
endif()

# MasterClient::FlushHeartbeat: immediate wakeup, no-op before start,
# flush-while-in-flight.
add_executable(test_umbp_master_client_flush_heartbeat
               test_master_client_flush_heartbeat.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_master_client_flush_heartbeat PRIVATE
                      -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_master_client_flush_heartbeat
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)
add_test(NAME umbp_master_client_flush_heartbeat
         COMMAND test_umbp_master_client_flush_heartbeat)

# Enables gtest_discover_tests() for the metadata-store / external-KV / SSD-tier
# suite migrated below from src/umbp/tests.
include(GoogleTest)

# MasterClient lifecycle: destructor budget, heartbeat thread shutdown, etc.
add_executable(test_umbp_master_client_lifecycle
               test_master_client_lifecycle.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_master_client_lifecycle PRIVATE
                      -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_master_client_lifecycle
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)
add_test(NAME umbp_master_client_lifecycle
         COMMAND test_umbp_master_client_lifecycle)

# MasterClient AddCounter / SetGauge / Observe buffering + PoolClient byte
# tracking integration.
add_executable(test_umbp_client_metrics test_umbp_client_metrics.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_client_metrics PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_client_metrics PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB}
                                   ${_TEST_GRPCPP_LIB} gtest)
add_test(NAME umbp_client_metrics COMMAND test_umbp_client_metrics)

# ScopedRpcTimer / MasterClient RPC latency instrumentation.
add_executable(test_umbp_master_client_rpc_latency
               test_master_client_rpc_latency.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_master_client_rpc_latency PRIVATE
                      -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_master_client_rpc_latency
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)
add_test(NAME umbp_master_client_rpc_latency
         COMMAND test_umbp_master_client_rpc_latency)

# BatchRouteGet columnar response wire-format round-trip (node-table dedup,
# 1-based node_ref, node_ref==0 -> not found, parallel tier/size columns). gRPC
# mock master + real MasterClient; no RDMA.
add_executable(test_umbp_batch_route_get_columnar
               test_batch_route_get_columnar.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_batch_route_get_columnar PRIVATE
                      -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_batch_route_get_columnar
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)
add_test(NAME umbp_batch_route_get_columnar
         COMMAND test_umbp_batch_route_get_columnar)

# Client-tag registration + metrics label injection.
add_executable(test_umbp_tags test_umbp_tags.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_umbp_tags PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_umbp_tags PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB}
                         ${_TEST_GRPCPP_LIB} gtest_main)
add_test(NAME umbp_tags COMMAND test_umbp_tags)

# --------------------------------------------------------------------------
# Unit tests (pure logic; no gRPC / RDMA).  Linked against umbp_common only.
# --------------------------------------------------------------------------

add_executable(test_umbp_types test_types.cpp)
target_link_libraries(test_umbp_types PRIVATE umbp_common gtest_main)
add_test(NAME umbp_types COMMAND test_umbp_types)

# env_time helpers are header-only; tests only exercise the parser path. NOTE:
# Production callers cache results in function-local statics that cannot be
# reset mid-process; any multi-value business test must fork.
add_executable(test_umbp_env_time test_env_time.cpp)
target_link_libraries(test_umbp_env_time PRIVATE umbp_common gtest_main)
add_test(NAME umbp_env_time COMMAND test_umbp_env_time)

add_executable(test_umbp_route_get_strategy test_route_get_strategy.cpp)
target_link_libraries(test_umbp_route_get_strategy PRIVATE umbp_common
                                                           gtest_main)
add_test(NAME umbp_route_get_strategy COMMAND test_umbp_route_get_strategy)

add_executable(test_umbp_route_put_strategy test_route_put_strategy.cpp)
target_link_libraries(test_umbp_route_put_strategy PRIVATE umbp_common
                                                           gtest_main)
add_test(NAME umbp_route_put_strategy COMMAND test_umbp_route_put_strategy)

# Master eviction policy plug-in: default LRU behavior + the EvictionManager DI
# seam.  The seam test calls EvictionManager::RunOnce() directly (a public
# single-pass trigger), so no special compile flags are needed.
add_executable(test_umbp_master_evict_strategy test_master_evict_strategy.cpp)
target_link_libraries(test_umbp_master_evict_strategy PRIVATE umbp_common
                                                              gtest_main)
add_test(NAME umbp_master_evict_strategy
         COMMAND test_umbp_master_evict_strategy)

# PageBitmapAllocator (peer-side per-tier page allocator primitive).
add_executable(test_umbp_page_bitmap_allocator test_page_bitmap_allocator.cpp)
target_link_libraries(test_umbp_page_bitmap_allocator PRIVATE umbp_common
                                                              gtest_main)
add_test(NAME umbp_page_bitmap_allocator
         COMMAND test_umbp_page_bitmap_allocator)

# --------------------------------------------------------------------------
# Integration tests (stand up in-process servers; gRPC + RDMA data path).
# Registered as CTest cases under the "integration" label.
# --------------------------------------------------------------------------

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

set_tests_properties(umbp_peer_service umbp_cross_node_smoke
                     umbp_pool_client_batch_put PROPERTIES LABELS "integration")

# --------------------------------------------------------------------------
# Benchmarks (manual; built but NOT registered with add_test).
# --------------------------------------------------------------------------

# PoolClient::BatchPut micro-bench (standalone executable, not a CTest target).
# Reports wall_ms / GiB/s; useful for integration regression tracking and NIC
# tuning.
add_executable(bench_umbp_pool_client_batch_put bench_pool_client_batch_put.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(bench_umbp_pool_client_batch_put PRIVATE
                      -Wl,--no-as-needed)
endif()
target_link_libraries(
  bench_umbp_pool_client_batch_put
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)

# PoolClient::BatchGet micro-bench (mirror of bench_pool_client_batch_put.cpp).
# Single implementation path; for ablation against an older revision, git
# checkout the previous release commit and re-run the binary built from that
# revision.
add_executable(bench_umbp_pool_client_batch_get bench_pool_client_batch_get.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(bench_umbp_pool_client_batch_get PRIVATE
                      -Wl,--no-as-needed)
endif()
target_link_libraries(
  bench_umbp_pool_client_batch_get
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)

# RoutePut strategy bench: compares batch put / put-then-get performance and
# placement distribution across UMBP_ROUTE_PUT_SELECT_ALGO x
# UMBP_ROUTE_PUT_NODE_AFFINITY under roomy/tight capacity regimes.  Standalone
# executable; not a CTest target.
add_executable(bench_umbp_route_put_strategy bench_route_put_strategy.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(bench_umbp_route_put_strategy PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(
  bench_umbp_route_put_strategy
  PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB} ${_TEST_GRPCPP_LIB}
          gtest_main)

# ===========================================================================
# Metadata-store / external-KV / SSD-tier suite (migrated from src/umbp/tests).
# These link umbp_common (pure logic) except test_peer_ssd_read_rpc, which
# exercises the gRPC peer service and links umbp_core + protobuf/gRPC. Test
# cases are registered individually via gtest_discover_tests.
# ===========================================================================

# test_master_metadata_store_interface — Phase 1 compile/instantiation gate for
# IMasterMetadataStore. Includes an isolated self-compile TU (proves the header
# is self-contained) and a GMock mock instantiation/signature-completeness test.
add_executable(
  test_master_metadata_store_interface test_master_metadata_store_interface.cpp
                                       master_metadata_store_self_compile.cpp)
target_include_directories(test_master_metadata_store_interface
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_master_metadata_store_interface
                      PRIVATE umbp_common GTest::gmock GTest::gtest_main)
target_compile_features(test_master_metadata_store_interface PRIVATE cxx_std_17)
gtest_discover_tests(test_master_metadata_store_interface)

# test_in_memory_master_metadata_store — Phase 2 behavioral suite for the
# InMemoryMasterMetadataStore implementation of IMasterMetadataStore (§6a).
add_executable(test_in_memory_master_metadata_store
               test_in_memory_master_metadata_store.cpp)
target_link_libraries(test_in_memory_master_metadata_store
                      PRIVATE umbp_common GTest::gtest_main)
target_compile_features(test_in_memory_master_metadata_store PRIVATE cxx_std_17)
gtest_discover_tests(test_in_memory_master_metadata_store)

# test_peer_dram_allocator
add_executable(test_peer_dram_allocator test_peer_dram_allocator.cpp)
target_link_libraries(test_peer_dram_allocator PRIVATE umbp_common
                                                       GTest::gtest_main)
target_compile_features(test_peer_dram_allocator PRIVATE cxx_std_17)
gtest_discover_tests(test_peer_dram_allocator)

# test_router_dedup — master-side BatchRoutePut dedup via IMasterMetadataStore
add_executable(test_router_dedup test_router_dedup.cpp)
target_link_libraries(test_router_dedup PRIVATE umbp_common GTest::gtest_main)
target_compile_features(test_router_dedup PRIVATE cxx_std_17)
gtest_discover_tests(test_router_dedup)

# test_peer_ssd_manager — SSD tier ownership + owned-location source
add_executable(test_peer_ssd_manager test_peer_ssd_manager.cpp)
target_link_libraries(test_peer_ssd_manager PRIVATE umbp_common
                                                    GTest::gtest_main)
target_compile_features(test_peer_ssd_manager PRIVATE cxx_std_17)
gtest_discover_tests(test_peer_ssd_manager)

# test_peer_ssd_eviction — LRU + watermark eviction + in-flight guard
add_executable(test_peer_ssd_eviction test_peer_ssd_eviction.cpp)
target_link_libraries(test_peer_ssd_eviction PRIVATE umbp_common
                                                     GTest::gtest_main)
target_compile_features(test_peer_ssd_eviction PRIVATE cxx_std_17)
gtest_discover_tests(test_peer_ssd_eviction)

# test_ssd_copy_pipeline — DramCopyPin + async copy-on-commit pipeline
add_executable(test_ssd_copy_pipeline test_ssd_copy_pipeline.cpp)
target_link_libraries(test_ssd_copy_pipeline PRIVATE umbp_common
                                                     GTest::gtest_main)
target_compile_features(test_ssd_copy_pipeline PRIVATE cxx_std_17)
gtest_discover_tests(test_ssd_copy_pipeline)

# test_tier_priority_route_get — RouteGet tier-priority strategy
add_executable(test_tier_priority_route_get test_tier_priority_route_get.cpp)
target_link_libraries(test_tier_priority_route_get PRIVATE umbp_common
                                                           GTest::gtest_main)
target_compile_features(test_tier_priority_route_get PRIVATE cxx_std_17)
gtest_discover_tests(test_tier_priority_route_get)

# test_peer_ssd_read_rpc — SSD read RPC over a gRPC loopback. Needs the gRPC
# peer service + generated stubs (umbp_core) and protobuf/gRPC.
add_executable(test_peer_ssd_read_rpc test_peer_ssd_read_rpc.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_peer_ssd_read_rpc PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(
  test_peer_ssd_read_rpc PRIVATE umbp_core umbp_common ${_TEST_PROTOBUF_LIB}
                                 ${_TEST_GRPCPP_LIB} GTest::gtest_main)
target_compile_features(test_peer_ssd_read_rpc PRIVATE cxx_std_17)
gtest_discover_tests(test_peer_ssd_read_rpc)

# test_ssd_read_lease_gating — pure reader-side lease gating decision logic
# (ssd_read_lease.h). Header-only under test (no gRPC / RDMA deps).
add_executable(test_ssd_read_lease_gating test_ssd_read_lease_gating.cpp)
target_link_libraries(test_ssd_read_lease_gating PRIVATE umbp_common
                                                         GTest::gtest_main)
target_compile_features(test_ssd_read_lease_gating PRIVATE cxx_std_17)
gtest_discover_tests(test_ssd_read_lease_gating)

# test_ssd_reliability — cross-component reliability: owned-source DRAM+SSD
# merge, SSD evict -> REMOVE -> master index convergence, tier-priority over the
# real index, crash-restart discard, and observability counters.
add_executable(test_ssd_reliability test_ssd_reliability.cpp)
target_link_libraries(test_ssd_reliability PRIVATE umbp_common
                                                   GTest::gtest_main)
target_compile_features(test_ssd_reliability PRIVATE cxx_std_17)
gtest_discover_tests(test_ssd_reliability)

# test_batch_resolve_codec — struct-of-arrays BatchResolveKeysResponse wire
# codec (batch_resolve_codec.h): encode/decode round-trip, page flattening, desc
# dedup / omit_descs, malformed-response rejection, and the payload-size win.
# Needs the generated proto (carried by umbp_common) + protobuf.
add_executable(test_batch_resolve_codec test_batch_resolve_codec.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_batch_resolve_codec PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(test_batch_resolve_codec
                      PRIVATE umbp_common ${_TEST_PROTOBUF_LIB} GTest::gtest_main)
target_compile_features(test_batch_resolve_codec PRIVATE cxx_std_17)
gtest_discover_tests(test_batch_resolve_codec)
