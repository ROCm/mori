# Device link stub — needed by any executable that links mori_shmem (which has
# -fgpu-rdc device code).  The .hip file triggers the HIP linker to resolve
# cross-TU device symbols like globalGpuStates.
set(DEVICE_LINK_STUB ${CMAKE_SOURCE_DIR}/src/pybind/device_link_stub.hip)
set(EXAMPLE_RDC_FLAGS "-fgpu-rdc" "-mcode-object-version=5")

# Helper: create an example executable that links mori_shmem (needs device link)
function(add_shmem_example name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
  add_executable(${name} ${ARG_SOURCES} ${DEVICE_LINK_STUB})
  target_link_libraries(${name} mori_shmem ${ARG_LIBS} hip::host hip::device)
  target_compile_options(${name} PRIVATE ${EXAMPLE_RDC_FLAGS})
  target_link_options(${name} PRIVATE ${EXAMPLE_RDC_FLAGS})
  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${name} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(ARG_INCLUDES)
    target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
  endif()
endfunction()

# --- SDMA examples ---
add_executable(sdma_bw sdma/sdma_bw.cpp sdma/common.cpp sdma/utils.cpp sdma/sdma_bw_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_bw PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_bw mori_application hip::host hip::device)

add_executable(sdma_rate sdma/sdma_rate.cpp sdma/common.cpp sdma/utils.cpp sdma/sdma_rate_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_rate PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_rate mori_application hip::host hip::device)

add_executable(sdma_latency sdma/sdma_latency.cpp sdma/common.cpp sdma/utils.cpp sdma/sdma_latency_kernel.cpp utils/args_parser.cpp)
target_include_directories(sdma_latency PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(sdma_latency mori_application hip::host hip::device)

# --- Dist RDMA examples ---
add_executable(dist_write dist_rdma_ops/dist_write.cpp utils/args_parser.cpp)
target_include_directories(dist_write PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
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

# --- Shmem examples (need device link for globalGpuStates) ---
add_shmem_example(put_thread_allgather       SOURCES shmem/put_thread_allgather.cpp)
add_shmem_example(concurrent_put_thread      SOURCES shmem/concurrent_put_thread.cpp)
add_shmem_example(test_put_schar_nbi_thread  SOURCES shmem/test_put_schar_nbi_thread.cpp)
add_shmem_example(concurrent_put_imm_thread  SOURCES shmem/concurrent_put_imm_thread.cpp)
add_shmem_example(concurrent_put_signal_thread SOURCES shmem/concurrent_put_signal_thread.cpp)
add_shmem_example(atomic_nonfetch_thread     SOURCES shmem/atomic_nonfetch_thread.cpp)
add_shmem_example(atomic_fetch_thread        SOURCES shmem/atomic_fetch_thread.cpp)
add_shmem_example(test_p2p_direct_access     SOURCES shmem/test_p2p_direct_access.cpp)

# --- Application examples ---
add_executable(context application/context.cpp)
target_link_libraries(context mori_application hip::host hip::device)

# --- Benchmarks ---
add_executable(atomic_perf benchmarks/atomic_perf.cpp utils/args_parser.cpp)
target_include_directories(atomic_perf PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(atomic_perf mori_application hip::host hip::device)

add_shmem_example(accum_perf SOURCES benchmarks/accum_perf.cpp)
add_shmem_example(ibverbs_test SOURCES application/ibverbs_test.cpp)

add_executable(test_vmm_mpi_write utils/test_vmm_mpi_write.cpp)
target_link_libraries(test_vmm_mpi_write mori_application hip::host hip::device)
