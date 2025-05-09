
# TODO: refactor 
# add_executable(
#     custom_allreduce
#     custom_allreduce/custom_allreduce.cpp
# )
# target_link_libraries(custom_allreduce mori_application hip::host hip::device)

# add_executable(
#     send_recv
#     local_rdma_ops/send_recv.cpp
# )
# target_link_libraries(send_recv mori_application hip::host hip::device)

add_executable(
    dist_read
    dist_rdma_ops/read.cpp
)
target_link_libraries(dist_read mori_application hip::host hip::device)

add_executable(
    send_recv_gpu
    local_rdma_ops/send_recv_gpu.cpp
)
target_link_libraries(send_recv_gpu mori_application hip::host hip::device)



add_executable(
    moe_all2all_dispatch
    moe_all2all/pplx-kernel-hip/internode_dispatch.cpp
)
target_link_libraries(moe_all2all_dispatch hip::host hip::device)

add_executable(
    put_warp
    shmem/put_warp.cpp
)
target_link_libraries(put_warp mori_shmem hip::host hip::device)