add_executable(dist_write dist_rdma_ops/dist_write.cpp utils/args_parser.cpp)
target_include_directories(dist_write PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
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

add_executable(concurrent_put_thread shmem/concurrent_put_thread.cpp)
target_link_libraries(concurrent_put_thread mori_shmem hip::host hip::device)

add_executable(concurrent_put_imm_thread shmem/concurrent_put_imm_thread.cpp)
target_link_libraries(concurrent_put_imm_thread mori_shmem hip::host
                      hip::device)

add_executable(concurrent_put_signal_thread shmem/concurrent_put_signal_thread.cpp)
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

add_executable(test_gpu_vmm utils/test_gpu_vmm.cpp)
target_link_libraries(test_gpu_vmm mori_application hip::host hip::device)

add_executable(test_vmm_mpi_write utils/test_vmm_mpi_write.cpp)
target_link_libraries(test_vmm_mpi_write mori_application hip::host hip::device)

add_executable(test_vmm_mpi_fd_direct utils/test_vmm_mpi_fd_direct.cpp)
target_link_libraries(test_vmm_mpi_fd_direct mori_application hip::host hip::device)

