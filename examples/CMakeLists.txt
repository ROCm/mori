add_executable(
    dist_read
    dist_rdma_ops/read.cpp
    utils/args_parser.cpp
)
target_include_directories(dist_read PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(dist_read mori_application hip::host hip::device)

add_executable(
    send_recv_gpu
    local_rdma_ops/send_recv_gpu.cpp
)
target_link_libraries(send_recv_gpu mori_application hip::host hip::device)

add_executable(
    write_inline_gpu
    local_rdma_ops/write_inline_gpu.cpp
)
target_link_libraries(write_inline_gpu mori_application hip::host hip::device)

add_executable(
    atomic_gpu
    local_rdma_ops/atomic_gpu.cpp
)
target_link_libraries(atomic_gpu mori_application hip::host hip::device)

add_executable(
    put_thread_allgather
    shmem/put_thread_allgather.cpp
)
target_link_libraries(put_thread_allgather mori_shmem hip::host hip::device)

add_executable(
    concurrent_put_thread
    shmem/concurrent_put_thread.cpp
)
target_link_libraries(concurrent_put_thread mori_shmem hip::host hip::device)

add_executable(
    context
    application/context.cpp
)
target_link_libraries(context mori_application hip::host hip::device)

add_executable(
    p2p_bandwidth
    benchmarks/p2p_bandwidth.cpp
)
target_link_libraries(p2p_bandwidth mori_application mori_shmem hip::host hip::device)

add_executable(
    test_dispatch_combine_ops
    ops/dispatch_combine/test_dispatch_combine.cpp
)
target_link_libraries(test_dispatch_combine_ops mori_application mori_shmem mori_ops hip::host hip::device)