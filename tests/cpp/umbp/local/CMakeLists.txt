# C++ unit tests
add_executable(test_dram_tier test_dram_tier.cpp)
target_link_libraries(test_dram_tier PRIVATE umbp_core)
add_test(NAME test_dram_tier COMMAND test_dram_tier)

add_executable(test_umbp_local_block_index test_local_block_index.cpp)
target_link_libraries(test_umbp_local_block_index PRIVATE umbp_core)
add_test(NAME umbp_local_block_index COMMAND test_umbp_local_block_index)

add_executable(test_umbp_local_storage_manager test_local_storage_manager.cpp)
target_link_libraries(test_umbp_local_storage_manager PRIVATE umbp_core)
add_test(NAME umbp_local_storage_manager
         COMMAND test_umbp_local_storage_manager)

add_executable(test_umbp_local_client test_umbp_client.cpp)
target_link_libraries(test_umbp_local_client PRIVATE umbp_core)
add_test(NAME umbp_local_client COMMAND test_umbp_local_client)

add_executable(test_umbp_local_follower_mode test_follower_mode.cpp)
target_link_libraries(test_umbp_local_follower_mode PRIVATE umbp_core)
add_test(NAME umbp_local_follower_mode COMMAND test_umbp_local_follower_mode)

add_executable(test_umbp_local_eviction test_prefix_aware_eviction.cpp)
target_link_libraries(test_umbp_local_eviction PRIVATE umbp_core)
add_test(NAME umbp_local_eviction COMMAND test_umbp_local_eviction)

add_executable(test_prefix_aware_eviction test_prefix_aware_eviction.cpp)
target_link_libraries(test_prefix_aware_eviction PRIVATE umbp_core)
add_test(NAME test_prefix_aware_eviction COMMAND test_prefix_aware_eviction)

add_executable(test_ssd_tier test_ssd_tier.cpp)
target_link_libraries(test_ssd_tier PRIVATE umbp_core)
add_test(NAME test_ssd_tier COMMAND test_ssd_tier)

# Micro-benchmark (manual run only, not registered as a test)
add_executable(bench_umbp_micro bench_umbp_micro.cpp)
target_link_libraries(bench_umbp_micro PRIVATE umbp_core)

# ---------------------------------------------------------------------------
# Helper: symlink spdk_proxy next to a target so SpawnProxyDaemon() finds it
# ---------------------------------------------------------------------------
function(umbp_symlink_spdk_proxy target)
  add_dependencies(${target} spdk_proxy)
  add_custom_command(
    TARGET ${target}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:spdk_proxy>
            $<TARGET_FILE_DIR:${target}>/spdk_proxy
    COMMENT "Symlink spdk_proxy next to ${target}")
endfunction()

# When spdk_proxy is available, symlink it next to targets that need it
if(TARGET spdk_proxy)
  umbp_symlink_spdk_proxy(bench_umbp_micro)

  # SPDK test/benchmark targets (require SPDK libraries at link time)
  add_executable(test_spdk_env test_spdk_env.cpp)
  target_link_libraries(test_spdk_env PRIVATE umbp_core)

  add_executable(test_spdk_ssd_tier test_spdk_ssd_tier.cpp)
  target_link_libraries(test_spdk_ssd_tier PRIVATE umbp_core)

  add_executable(bench_spdk_raw bench_spdk_raw.cpp)
  target_link_libraries(bench_spdk_raw PRIVATE umbp_core)
endif()

# Unix-only targets (SHM IPC / fork-based, no SPDK dependency)
if(UNIX)
  add_executable(bench_proxy_ipc bench_proxy_ipc.cpp)
  target_link_libraries(bench_proxy_ipc PRIVATE umbp_core)

  add_executable(bench_umbp_e2e bench_umbp_e2e.cpp)
  target_link_libraries(bench_umbp_e2e PRIVATE umbp_core)

  add_executable(test_umbp_e2e test_umbp_e2e.cpp)
  target_link_libraries(test_umbp_e2e PRIVATE umbp_core)

  if(TARGET spdk_proxy)
    umbp_symlink_spdk_proxy(bench_proxy_ipc)
    umbp_symlink_spdk_proxy(bench_umbp_e2e)
    umbp_symlink_spdk_proxy(test_umbp_e2e)
  endif()
endif()
