# Shmem examples contain device code (__global__ kernels) and must be compiled
# with hipcc. They define their own globalGpuStates. After ShmemInit, they call
# ShmemModuleInit on the loaded JIT module to get GpuStates initialized.

# Helper: create a shmem example (HIP compiled, links mori_shmem)
function(add_shmem_example name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
  add_executable(${name} ${ARG_SOURCES})
  set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE HIP)
  target_link_libraries(${name} mori_shmem ${ARG_LIBS} hip::host hip::device)
  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${name} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(ARG_INCLUDES)
    target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
  endif()
endfunction()

# --- SDMA examples ---
add_executable(sdma_bw sdma/sdma_bw.cpp sdma/common.cpp sdma/utils.cpp
                       sdma/sdma_bw_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_bw PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_bw mori_application hip::host hip::device)

add_executable(
  sdma_bw_allgather sdma/sdma_bw_allgather.cpp sdma/common.cpp sdma/utils.cpp
                    sdma/sdma_bw_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_bw_allgather
                           PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_bw_allgather mori_application hip::host hip::device)

add_executable(sdma_rate sdma/sdma_rate.cpp sdma/common.cpp sdma/utils.cpp
                         sdma/sdma_rate_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_rate PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_rate mori_application hip::host hip::device)

add_executable(
  sdma_latency sdma/sdma_latency.cpp sdma/common.cpp sdma/utils.cpp
               sdma/sdma_latency_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_latency
                           PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_latency mori_application hip::host hip::device)

# --- Dist RDMA examples ---
add_executable(dist_write dist_rdma_ops/dist_write.cpp utils/args_parser.cpp)
target_include_directories(dist_write
                           PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(dist_write mori_application hip::host hip::device)

# --- Local RDMA examples ---
add_executable(send_recv_gpu local_rdma_ops/send_recv_gpu.cpp)
target_link_libraries(send_recv_gpu mori_application hip::host hip::device)

add_executable(write_gpu local_rdma_ops/write_gpu.cpp)
target_link_libraries(write_gpu mori_application hip::host hip::device)

add_executable(write_inline_gpu local_rdma_ops/write_inline_gpu.cpp)
target_link_libraries(write_inline_gpu mori_application hip::host hip::device)

add_executable(atomic_gpu local_rdma_ops/atomic_gpu.cpp)
target_link_libraries(atomic_gpu mori_application hip::host hip::device)

# --- Shmem examples (HIP compiled, contain device kernels) ---
add_shmem_example(put_thread_allgather SOURCES shmem/put_thread_allgather.cpp)
add_shmem_example(allgather_test SOURCES shmem/allgather_test.cpp)
add_shmem_example(concurrent_put_thread SOURCES shmem/concurrent_put_thread.cpp)
add_shmem_example(concurrent_get_thread SOURCES shmem/concurrent_get_thread.cpp)
add_shmem_example(test_put_schar_nbi_thread SOURCES
                  shmem/test_put_schar_nbi_thread.cpp)
add_shmem_example(concurrent_put_imm_thread SOURCES
                  shmem/concurrent_put_imm_thread.cpp)
add_shmem_example(concurrent_put_signal_thread SOURCES
                  shmem/concurrent_put_signal_thread.cpp)
add_shmem_example(atomic_nonfetch_thread SOURCES
                  shmem/atomic_nonfetch_thread.cpp)
add_shmem_example(atomic_fetch_thread SOURCES shmem/atomic_fetch_thread.cpp)
add_shmem_example(test_p2p_direct_access SOURCES
                  shmem/test_p2p_direct_access.cpp)
add_shmem_example(sdma_put_test SOURCES shmem/sdma_put_test.cpp)
add_shmem_example(multithread_multi_gpu SOURCES
                  shmem/multithread_multi_gpu.cpp)

# --- Collective examples ---
add_executable(all_reduce collective/inter_node/oneshot_example.cpp)
target_link_libraries(all_reduce mori_collective hip::host hip::device)

add_executable(all2all collective/inter_node/oneshot_all2all.cpp)
target_link_libraries(all2all mori_collective hip::host hip::device)

add_executable(all_gather collective/inter_node/all_gather_example.cpp)
target_link_libraries(all_gather mori_shmem hip::host hip::device)

add_executable(allgather_sdma collective/sdma_allgather_example.cpp)
target_link_libraries(allgather_sdma mori_collective mori_shmem hip::host
                      hip::device)

add_executable(all2all_sdma collective/sdma_all2all_example.cpp)
target_link_libraries(all2all_sdma mori_collective mori_shmem hip::host
                      hip::device)

add_executable(allreduce_sdma collective/sdma_allreduce_example.cpp)
target_link_libraries(allreduce_sdma mori_collective mori_shmem hip::host
                      hip::device)

add_executable(allreduce_sdma_sync collective/sdma_allreduce_sync_example.cpp)
target_link_libraries(allreduce_sdma_sync mori_collective mori_shmem hip::host
                      hip::device)
add_executable(shmem_allreduce collective/shmem_allreduce_example.cpp)
target_link_libraries(shmem_allreduce mori_collective mori_shmem hip::host
                      hip::device)
add_executable(sdma_self_copy_test sdma/sdma_self_copy_test.cpp)
target_link_libraries(sdma_self_copy_test mori_collective mori_shmem hip::host
                      hip::device)

# Find MPI package (required by intra-node examples)
find_package(MPI REQUIRED)

add_executable(allreduce_example collective/intra_node/intra_node_example.cpp)
target_link_libraries(allreduce_example mori_collective MPI::MPI_CXX hip::host
                      hip::device)
target_include_directories(allreduce_example
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(intra_node_benchmark
               collective/intra_node/intra_node_benchmark.cpp)
target_link_libraries(intra_node_benchmark mori_collective MPI::MPI_CXX
                      hip::host hip::device)
target_include_directories(intra_node_benchmark
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

# --- Application examples ---
add_executable(context application/context.cpp)
target_link_libraries(context mori_application hip::host hip::device)

# --- Benchmarks ---
add_executable(atomic_perf benchmarks/atomic_perf.cpp utils/args_parser.cpp)
target_include_directories(atomic_perf
                           PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(atomic_perf mori_application hip::host hip::device)

add_shmem_example(accum_perf SOURCES benchmarks/accum_perf.cpp)
add_shmem_example(ibverbs_test SOURCES application/ibverbs_test.cpp)

add_executable(test_vmm_mpi_write utils/test_vmm_mpi_write.cpp)
target_link_libraries(test_vmm_mpi_write mori_application hip::host hip::device)
