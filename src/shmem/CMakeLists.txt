find_package(MPI REQUIRED)

add_library(mori_shmem init.cpp memory.cpp)
target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
  mori_shmem
  mori_application
  MPI::MPI_CXX
  ibverbs
  hip::host
  hip::device
  mlx5)
target_compile_options(mori_shmem PUBLIC "-fgpu-rdc")
target_link_options(mori_shmem PUBLIC "-fgpu-rdc")
