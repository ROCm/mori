find_package(MPI REQUIRED)

set(MORI_SHMEM_SOURCES init.cpp memory.cpp)

# All shmem source files are now pure CXX (device code moved to JIT)
add_library(mori_shmem ${MORI_SHMEM_SOURCES})

target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
  mori_shmem
  PUBLIC mori_application mori_logging MPI::MPI_CXX ibverbs hip::host mlx5)
