find_package(MPI REQUIRED)


add_library(
    mori_application
    bootstrap/mpi_bootstrap.cpp
    transport/rdma/rdma.cpp
    transport/rdma/providers/mlx5/mlx5.cpp
    memory/symmetric_memory.cpp
    memory/memory_region.cpp
)

target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mori_application MPI::MPI_CXX ibverbs hip::host hip::device mlx5)