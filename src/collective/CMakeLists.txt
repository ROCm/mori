# Source files list
set(COLLECTIVE_SOURCES
    core/oneshot_all2all_sdma_class.cpp core/oneshot_allgather_sdma_class.cpp
    core/twoshot_allreduce_sdma_class.cpp
    # Note: intra_node_executor is header-only template class
)

if(WITH_MPI)
  find_package(MPI REQUIRED)
  list(APPEND COLLECTIVE_SOURCES core/topology_detector.cpp
       core/allreduce_manager.cpp inter_node/executors/ring_1d_executor.cpp
       inter_node/executors/one_shot_executor.cpp)
endif()

add_library(mori_collective SHARED ${COLLECTIVE_SOURCES})

# HIP compile options (extracted from aiter, PyTorch dependencies removed)
if(USE_ROCM)
  # NO_STATIC_INIT: skip auto barrier launcher from shmem.hpp.
  # ENABLE_WEAK_GLOBAL_GPU_STATES collective HIP TUs need weak globalGpuStates +
  # RegisterGpuStatesAddrProvider; EP JIT must not set this (uses
  # MORI_DEFINE_GPU_STATES). See include/mori/shmem/shmem.hpp "Policy: EP vs
  # SDMA CCL".
  target_compile_definitions(
    mori_collective
    PRIVATE __HIP_PLATFORM_AMD__=1 USE_ROCM=1 HIPBLAS_V2 __HIP_PLATFORM_HCC__=1
            MORI_SHMEM_NO_STATIC_INIT MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES)

  # HIP-specific compile options
  target_compile_options(
    mori_collective
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

target_link_libraries(
  mori_collective
  PUBLIC mori_shmem mori_application mori_logging
  PRIVATE hip::host hip::device)

if(WITH_MPI)
  target_link_libraries(mori_collective PUBLIC MPI::MPI_CXX)
endif()

target_include_directories(mori_collective PUBLIC ${CMAKE_SOURCE_DIR}/include)
