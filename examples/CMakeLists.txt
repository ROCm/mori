add_executable(sdma_bw sdma/sdma_bw.cpp sdma/common.cpp sdma/utils.cpp sdma/sdma_bw_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_bw PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_bw mori_application hip::host hip::device)

add_executable(sdma_bw_allgather sdma/sdma_bw_allgather.cpp sdma/common.cpp sdma/utils.cpp sdma/sdma_bw_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_bw_allgather PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_bw_allgather mori_application hip::host hip::device)

add_executable(sdma_rate sdma/sdma_rate.cpp sdma/common.cpp sdma/utils.cpp sdma/sdma_rate_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_rate PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_rate mori_application hip::host hip::device)

add_executable(sdma_latency sdma/sdma_latency.cpp sdma/common.cpp sdma/utils.cpp sdma/sdma_latency_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_latency PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_latency mori_application hip::host hip::device)

# Find MPI package (required by allreduce_example)
find_package(MPI REQUIRED)

add_executable(dist_write dist_rdma_ops/dist_write.cpp utils/args_parser.cpp)
target_include_directories(dist_write
                           PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(dist_write mori_application hip::host hip::device)

add_executable(send_recv_gpu local_rdma_ops/send_recv_gpu.cpp)
target_link_libraries(send_recv_gpu mori_application hip::host hip::device)

add_executable(write_gpu local_rdma_ops/write_gpu.cpp)
target_link_libraries(write_gpu mori_application hip::host hip::device)

add_executable(write_inline_gpu local_rdma_ops/write_inline_gpu.cpp)
target_link_libraries(write_inline_gpu mori_application hip::host hip::device)

add_executable(atomic_gpu local_rdma_ops/atomic_gpu.cpp)
target_link_libraries(atomic_gpu mori_application hip::host hip::device)

add_executable(put_thread_allgather shmem/put_thread_allgather.cpp)
target_link_libraries(put_thread_allgather mori_shmem hip::host hip::device)

add_executable(all_reduce collective/inter_node/oneshot_example.cpp)
target_link_libraries(all_reduce mori_collective hip::host hip::device)

add_executable(all2all collective/inter_node/oneshot_all2all.cpp)
target_link_libraries(all2all mori_collective hip::host hip::device)

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

add_executable(all_gather collective/inter_node/all_gather_example.cpp)
target_link_libraries(all_gather mori_shmem hip::host hip::device)

add_executable(allgather_sdma collective/sdma_allgather_example.cpp)
target_link_libraries(allgather_mori_collective sdma mori_shmem hip::host hip::device)
add_executable(all2all_sdma collective/sdma_all2all_example.cpp)
target_link_libraries(all2all_sdma mori_collective mori_shmem hip::host hip::device)

add_executable(concurrent_put_thread shmem/concurrent_put_thread.cpp)
target_link_libraries(concurrent_put_thread mori_shmem hip::host hip::device)

add_executable(concurrent_put_imm_thread shmem/concurrent_put_imm_thread.cpp)
target_link_libraries(concurrent_put_imm_thread mori_shmem hip::host
                      hip::device)

add_executable(concurrent_put_signal_thread
               shmem/concurrent_put_signal_thread.cpp)
target_link_libraries(concurrent_put_signal_thread mori_shmem hip::host
                      hip::device)

add_executable(atomic_nonfetch_thread shmem/atomic_nonfetch_thread.cpp)
target_link_libraries(atomic_nonfetch_thread mori_shmem hip::host hip::device)

add_executable(atomic_fetch_thread shmem/atomic_fetch_thread.cpp)
target_link_libraries(atomic_fetch_thread mori_shmem hip::host hip::device)

add_executable(context application/context.cpp)
target_link_libraries(context mori_application hip::host hip::device)

add_executable(test_dispatch_combine_ops
               ops/dispatch_combine/test_dispatch_combine.cpp)
target_link_libraries(test_dispatch_combine_ops mori_application mori_shmem
                      mori_ops hip::host hip::device)

add_executable(atomic_perf benchmarks/atomic_perf.cpp utils/args_parser.cpp)
target_include_directories(atomic_perf
                           PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(atomic_perf mori_application hip::host hip::device)

add_executable(accum_perf benchmarks/accum_perf.cpp)
target_link_libraries(accum_perf mori_application mori_shmem hip::host
                      hip::device)

add_executable(ibverbs_test application/ibverbs_test.cpp)
target_link_libraries(ibverbs_test mori_application mori_shmem hip::host
                      hip::device)
