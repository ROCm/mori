# Source files list
set(COLLECTIVE_SOURCES
    core/oneshot_all2all_sdma_class.cpp core/oneshot_allgather_sdma_class.cpp
    core/allgather_into_tensor.cpp core/twoshot_allreduce_sdma_class.cpp
    # Note: intra_node_executor is header-only template class
)

add_library(mori_collective SHARED ${COLLECTIVE_SOURCES})

if(USE_ROCM)
  target_compile_definitions(
    mori_collective PRIVATE __HIP_PLATFORM_AMD__=1 USE_ROCM=1 HIPBLAS_V2
                            __HIP_PLATFORM_HCC__=1 MORI_SHMEM_NO_STATIC_INIT)
endif()

# Host-only: SDMA kernels are JIT-compiled from ccl_kernels.hip at runtime. No
# hip::device (no -x hip, no --hip-link) — avoids atexit/glibc compat issues.
target_link_libraries(
  mori_collective
  PUBLIC mori_shmem mori_application mori_logging
  PRIVATE hip::host)

# Inter-node MPI executors contain AOT device code (<<<>>> launches) and require
# hip::device. Built as a separate object library so the core SDMA path stays
# host-only.
if(WITH_MPI)
  find_package(MPI REQUIRED)
  add_library(
    mori_collective_internode OBJECT
    core/topology_detector.cpp core/allreduce_manager.cpp
    inter_node/executors/ring_1d_executor.cpp
    inter_node/executors/one_shot_executor.cpp)
  target_include_directories(mori_collective_internode
                             PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(mori_collective_internode PUBLIC MPI::MPI_CXX hip::host
                                                         hip::device)
  if(USE_ROCM)
    target_compile_definitions(
      mori_collective_internode
      PRIVATE __HIP_PLATFORM_AMD__=1 USE_ROCM=1 HIPBLAS_V2
              __HIP_PLATFORM_HCC__=1 MORI_SHMEM_NO_STATIC_INIT
              MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES)
    target_compile_options(
      mori_collective_internode
      PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
              --offload-arch=native
              -fgpu-flush-denormals-to-zero
              -fno-offload-uniform-block
              -mllvm
              --amdgpu-kernarg-preload-count=16
              -fno-gpu-rdc
              -Wno-unused-result
              >)
  endif()
  target_link_libraries(mori_collective PRIVATE mori_collective_internode
                                                MPI::MPI_CXX)
endif()

target_include_directories(mori_collective PUBLIC ${CMAKE_SOURCE_DIR}/include)
