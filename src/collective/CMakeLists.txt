find_package(MPI REQUIRED)

add_library(
  mori_collective SHARED allreduce/allreduce_manager.cpp topology_detector.cpp
                         allreduce/ring_1d_executor.cpp)

target_link_libraries(
  mori_collective
  PUBLIC mori_shmem mori_application mori_logging MPI::MPI_CXX
  PRIVATE hip::host hip::device)

target_include_directories(mori_collective PUBLIC ${CMAKE_SOURCE_DIR}/include)
