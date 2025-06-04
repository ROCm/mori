find_package(MPI REQUIRED)
# find_package(Torch REQUIRED)

add_library(
    mori_application
    bootstrap/mpi_bootstrap.cpp
    # bootstrap/torch_bootstrap.cpp
    transport/rdma/rdma.cpp
    transport/rdma/providers/mlx5/mlx5.cpp
    memory/symmetric_memory.cpp
    memory/memory_region.cpp
    context/context.cpp
)

target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mori_application MPI::MPI_CXX ibverbs hip::host hip::device mlx5)

# target_include_directories(mori_application PUBLIC ${TORCH_INCLUDE_DIRS})
# target_link_libraries(mori_application INTERFACE c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart)