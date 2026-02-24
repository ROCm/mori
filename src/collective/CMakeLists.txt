find_package(MPI REQUIRED)

# Source files list
set(COLLECTIVE_SOURCES
    core/topology_detector.cpp
    core/allreduce_manager.cpp
	core/oneshot_all2all_sdma_class.cpp
    core/oneshot_allgather_sdma_class.cpp
    core/twoshot_allreduce_sdma_class.cpp
    inter_node/executors/ring_1d_executor.cpp
    inter_node/executors/one_shot_executor.cpp
    # Note: intra_node_executor is header-only template class
)

add_library(mori_collective SHARED ${COLLECTIVE_SOURCES})

# HIP compile options (extracted from aiter, PyTorch dependencies removed)
if(USE_ROCM)
  target_compile_definitions(
    mori_collective PRIVATE __HIP_PLATFORM_AMD__=1 USE_ROCM=1 HIPBLAS_V2
                            __HIP_PLATFORM_HCC__=1)

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
  PUBLIC mori_shmem mori_application mori_logging MPI::MPI_CXX
  PRIVATE hip::host hip::device)

target_include_directories(mori_collective PUBLIC ${CMAKE_SOURCE_DIR}/include)
