find_package(MPI REQUIRED)

execute_process(
  COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE TORCH_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
cmake_path(SET TORCH_CMAKE_DIR NORMALIZE "${TORCH_DIR}/Torch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_DIR})
message(STATUS "Found LibTorch CMake Path: ${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)

add_library(
  mori_application SHARED
  bootstrap/mpi_bootstrap.cpp
  bootstrap/torch_bootstrap.cpp
  bootstrap/local_bootstrap.cpp
  transport/rdma/rdma.cpp
  transport/rdma/providers/mlx5/mlx5.cpp
  transport/rdma/providers/bnxt/bnxt.cpp
  transport/rdma/providers/ibverbs/ibverbs.cpp
  transport/tcp/tcp.cpp
  memory/symmetric_memory.cpp
  memory/vmm_va_manager.cpp
  memory/memory_region.cpp
  context/context.cpp
  topology/gpu.cpp
  topology/net.cpp
  topology/node.cpp
  topology/pci.cpp
  topology/system.cpp)

target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
  mori_application
  MPI::MPI_CXX
  ibverbs
  hip::host
  hip::device
  mlx5
  rocm_smi64
  pci
  mori_logging)

if(USE_BNXT)
  target_link_libraries(mori_application ${BNXT_RE_LIB})
endif()

target_include_directories(mori_application PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(mori_application c10 torch torch_cpu c10_hip torch_hip)
