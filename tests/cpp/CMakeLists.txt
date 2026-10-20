# ---------------------------------------------------------------------------
# GoogleTest — fetched once here; gtest_main is available to all subdirs.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(googletest)

add_library(test_sdma_allreduce_public_header_compile OBJECT
            collective/test_sdma_allreduce_public_header.hip)
set_source_files_properties(
  collective/test_sdma_allreduce_public_header.hip PROPERTIES LANGUAGE HIP)
target_include_directories(test_sdma_allreduce_public_header_compile
                           PRIVATE ${CMAKE_SOURCE_DIR}/include
                                   ${CMAKE_SOURCE_DIR})
if(DEFINED GPU_TARGETS)
  set_target_properties(
    test_sdma_allreduce_public_header_compile
    PROPERTIES HIP_ARCHITECTURES "${GPU_TARGETS}")
endif()

# WarpAccum / WarpAccumLF correctness sweep + micro-benchmark on a single GPU
# (header-only, no MPI/shmem). Needs a GPU, so it is not registered with ctest.
add_executable(test_warp_accum ops/test_warp_accum.cpp)
set_source_files_properties(ops/test_warp_accum.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(test_warp_accum PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                   ${CMAKE_SOURCE_DIR})
target_link_libraries(test_warp_accum hip::host hip::device)
# Match the kernel hipcc line (src/ops/CMakeLists.txt) so line tables in the
# disassembly point at device_primitives.hpp and assert() stays compiled out.
target_compile_options(test_warp_accum PRIVATE -gline-tables-only -DNDEBUG)
if(DEFINED GPU_TARGETS)
  set_target_properties(test_warp_accum PROPERTIES HIP_ARCHITECTURES
                                                   "${GPU_TARGETS}")
  # Same as the JIT build: HW fp4/fp8 converts on gfx950. The define cannot be
  # set per arch, so only add it when gfx950 is the sole target.
  if(GPU_TARGETS STREQUAL "gfx950")
    target_compile_definitions(test_warp_accum
                               PRIVATE HIP_ENABLE_GFX950_OCP_BUILTINS=1)
  endif()
endif()

# Two-GPU BlockCopy alignment sweep (P2P via shmem). Needs MPI and two GPUs,
# so it is not registered with ctest.
if(BUILD_SHMEM AND WITH_MPI)
  find_package(MPI REQUIRED)
  add_executable(test_block_copy ops/test_block_copy.cpp)
  set_source_files_properties(ops/test_block_copy.cpp PROPERTIES LANGUAGE HIP)
  target_include_directories(
    test_block_copy PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}
                            ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(test_block_copy mori_shmem hip::host hip::device
                        ${MPI_CXX_LIBRARIES})
  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(test_block_copy PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(DEFINED GPU_TARGETS)
    set_target_properties(test_block_copy PROPERTIES HIP_ARCHITECTURES
                                                     "${GPU_TARGETS}")
  endif()
endif()

# Core logging is header-only (spdlog); test its thread-safety independently of
# BUILD_IO. Run under ThreadSanitizer to actually prove the maps are safe.
add_executable(test_logging utils/test_logging.cpp)
target_include_directories(test_logging PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_logging PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_logging PRIVATE mori_logging gtest_main)
add_test(NAME logging_concurrency COMMAND test_logging)

# Header-only reader for flat YAML lists (mori/utils/flat_yaml.hpp).
add_executable(test_flat_yaml utils/test_flat_yaml.cpp)
target_include_directories(test_flat_yaml PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_flat_yaml PRIVATE gtest_main)
add_test(NAME flat_yaml COMMAND test_flat_yaml)

# PCI topology over a sysfs tree built on disk; needs no hardware.
add_executable(test_topology_pci application/test_topology_pci.cpp)
target_include_directories(test_topology_pci PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_topology_pci PRIVATE mori_application gtest_main)
add_test(NAME topology_pci COMMAND test_topology_pci)

add_executable(test_transport_tcp application/test_transport_tcp.cpp)

target_include_directories(test_transport_tcp
                           PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_transport_tcp PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_transport_tcp mori_application)

add_executable(test_transport_ibverbs application/test_transport_ibverbs.cpp)

target_include_directories(test_transport_ibverbs
                           PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_transport_ibverbs PUBLIC ${CMAKE_SOURCE_DIR})
# This test calls libibverbs directly (e.g. ibv_get_cq_event); mori_application
# no longer re-exports those symbols (they are dlopen'd internally). Link the
# dlopen shim instead of the real libibverbs so BUILD_TESTS=ON does not
# reintroduce a link-time dependency on a host without RDMA installed.
target_link_libraries(test_transport_ibverbs mori_application mori_ibv_shim)

# Pure host planner: no HIP runtime, IO library, or GPU required.
add_executable(test_fabric_copy_plan io/test_fabric_copy_plan.cpp
               ${CMAKE_SOURCE_DIR}/src/io/fabric/copy_plan.cpp)
target_include_directories(test_fabric_copy_plan PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME fabric_copy_plan COMMAND test_fabric_copy_plan)

if(BUILD_IO)
  add_executable(test_fabric_batch_copy io/test_fabric_batch_copy.cpp)
  target_link_libraries(test_fabric_batch_copy PRIVATE mori_io)
  # Deliberately not a ctest: requires a JIT-produced hsaco and fabric GPUs.

  add_executable(test_protocol io/test_protocol.cpp)

  target_include_directories(test_protocol PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_protocol PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_protocol mori_application mori_io)

  add_executable(test_rail_affinity io/test_rail_affinity.cpp)

  target_include_directories(test_rail_affinity
                             PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_rail_affinity PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_rail_affinity mori_application mori_io numa)

  add_executable(test_engine io/test_engine.cpp)

  target_include_directories(test_engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_engine PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_engine mori_application mori_io)

  add_executable(bench_engine io/bench_engine.cpp)

  target_include_directories(bench_engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(bench_engine PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(bench_engine mori_application mori_io)

  add_executable(test_transfer_wait io/test_transfer_wait.cpp)

  target_include_directories(test_transfer_wait
                             PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_transfer_wait PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_transfer_wait mori_application mori_io)

  if(ENABLE_IO_FAULT_INJECTION)
    add_executable(test_fault_injection io/test_fault_injection.cpp)
    target_include_directories(test_fault_injection
                               PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(test_fault_injection PUBLIC ${CMAKE_SOURCE_DIR})
    target_compile_definitions(
      test_fault_injection
      PRIVATE
        MORI_IO_FAULT_CATALOG_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/io/fault_catalog/ibverbs.yaml"
        MORI_IO_FAULT_CATALOG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/io/fault_catalog"
    )
    target_link_libraries(test_fault_injection mori_application mori_io)

    # CPU-only regression checks with provider stand-ins; no libibverbs link.
    add_executable(test_fault_injector io/test_fault_injector.cpp
                   ${CMAKE_SOURCE_DIR}/src/io/rdma/fault_injector.cpp)
    target_include_directories(test_fault_injector PRIVATE ${CMAKE_SOURCE_DIR}
                               ${CMAKE_SOURCE_DIR}/include
                               ${CMAKE_SOURCE_DIR}/3rdparty/msgpack-c/include)
    target_compile_definitions(test_fault_injector PRIVATE MORI_IO_FAULT_INJECTION
                                                          MSGPACK_NO_BOOST)
    target_link_libraries(test_fault_injector mori_logging hip::host)
  endif()

  add_executable(test_peer_failure io/test_peer_failure.cpp)

  target_include_directories(test_peer_failure PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_peer_failure PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_peer_failure mori_application mori_io)

  add_executable(test_fence io/test_fence.cpp)

  target_include_directories(test_fence PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_fence PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_fence mori_application mori_io)

  add_executable(test_fence_xnode io/test_fence_xnode.cpp)

  target_include_directories(test_fence_xnode PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_fence_xnode PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_fence_xnode mori_application mori_io)

  add_executable(test_topology application/test_topology.cpp)

  target_include_directories(test_topology PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_topology PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_topology mori_application mori_io)

  add_executable(test_discrete_buffer io/test_discrete_buffer.cpp)
  set_source_files_properties(io/test_discrete_buffer.cpp PROPERTIES LANGUAGE
                                                                     HIP)

  target_include_directories(test_discrete_buffer
                             PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_discrete_buffer PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_discrete_buffer hip::host hip::device)
  if(DEFINED GPU_TARGETS)
    set_target_properties(test_discrete_buffer PROPERTIES HIP_ARCHITECTURES
                                                          "${GPU_TARGETS}")
  endif()

  # OFI/Slingshot correctness test — needs MPI to launch 2 ranks.
  if(WITH_MPI)
    find_package(MPI REQUIRED)
    add_executable(ofi_slingshot_test io/ofi_slingshot_test.cpp)
    target_include_directories(ofi_slingshot_test
                               PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(ofi_slingshot_test PUBLIC ${CMAKE_SOURCE_DIR})
    target_include_directories(ofi_slingshot_test
                               PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(ofi_slingshot_test mori_application mori_io
                          ${MPI_CXX_LIBRARIES})
  endif()
endif()

if(BUILD_OPS_DEVICE AND WITH_MPI)
  find_package(MPI REQUIRED)
  add_executable(test_aot_launch ops/test_aot_launch.cpp)
  target_include_directories(test_aot_launch PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(test_aot_launch PUBLIC ${CMAKE_SOURCE_DIR})
  target_include_directories(test_aot_launch PUBLIC ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(test_aot_launch mori_ops mori_shmem mori_application
                        hip::host ${MPI_CXX_LIBRARIES})
endif()

if(BUILD_CCO)
  add_subdirectory(cco)
endif()

if(BUILD_UMBP)
  add_subdirectory(umbp)
endif()

if(BUILD_METRICS)
  add_subdirectory(metrics)
endif()

add_subdirectory(ras)

# JIT v2 — always built: mori_jit is host-only and needs no hipcc.
add_subdirectory(jit)
