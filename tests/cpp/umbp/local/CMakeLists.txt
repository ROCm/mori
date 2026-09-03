# Storage-tier unit tests.
#
# These cover umbp_core's tier layer -- the SSD/SPDK/segment backends and the
# host allocator -- which outlived the local client that used to sit on top of
# them: PeerSsdManager builds the distributed SSD medium on exactly these
# classes.  The tests that drove StandaloneClient, LocalStorageManager,
# LocalBlockIndex and DRAMTier went with those types; DRAM is now covered by the
# PageBackend tests under ../distributed.
add_executable(test_umbp_host_mem_allocator test_host_mem_allocator.cpp)
target_link_libraries(test_umbp_host_mem_allocator PRIVATE umbp_core)
add_test(NAME umbp_host_mem_allocator COMMAND test_umbp_host_mem_allocator)

add_executable(test_ssd_tier test_ssd_tier.cpp)
target_link_libraries(test_ssd_tier PRIVATE umbp_core)
add_test(NAME test_ssd_tier COMMAND test_ssd_tier)

add_executable(test_umbp_ssd_tier_ranges test_ssd_tier_ranges.cpp)
target_link_libraries(test_umbp_ssd_tier_ranges PRIVATE umbp_core GTest::gtest_main)
target_compile_features(test_umbp_ssd_tier_ranges PRIVATE cxx_std_17)
add_test(NAME umbp_ssd_tier_ranges COMMAND test_umbp_ssd_tier_ranges)

add_executable(test_segment_crc test_segment_crc.cpp)
target_link_libraries(test_segment_crc PRIVATE umbp_core)
add_test(NAME test_segment_crc COMMAND test_segment_crc)

add_executable(test_sharded_ssd_tier test_sharded_ssd_tier.cpp)
target_link_libraries(test_sharded_ssd_tier PRIVATE umbp_core)
add_test(NAME test_sharded_ssd_tier COMMAND test_sharded_ssd_tier)

add_executable(test_ssd_direct_io test_ssd_direct_io.cpp)
target_link_libraries(test_ssd_direct_io PRIVATE umbp_core)
add_test(NAME test_ssd_direct_io COMMAND test_ssd_direct_io)

add_executable(test_dummy_ssd_tier test_dummy_ssd_tier.cpp)
target_link_libraries(test_dummy_ssd_tier PRIVATE umbp_core)
add_test(NAME test_dummy_ssd_tier COMMAND test_dummy_ssd_tier)

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

  if(TARGET spdk_proxy)
    umbp_symlink_spdk_proxy(bench_proxy_ipc)
  endif()
endif()
