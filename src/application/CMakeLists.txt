find_package(MPI REQUIRED)
find_package(hsa-runtime64 REQUIRED)
find_package(hsakmt REQUIRED)

# ---- Core application library (no Torch dependency) ----
add_library(
  mori_application_core SHARED
  bootstrap/mpi_bootstrap.cpp
  bootstrap/socket_bootstrap.cpp
  bootstrap/local_bootstrap.cpp
  transport/rdma/rdma.cpp
  transport/rdma/providers/mlx5/mlx5.cpp
  transport/rdma/providers/bnxt/bnxt.cpp
  transport/rdma/providers/ionic/ionic.cpp
  transport/rdma/providers/ibverbs/ibverbs.cpp
  transport/tcp/tcp.cpp
  transport/sdma/anvil.cpp
  memory/symmetric_memory.cpp
  memory/va_manager.cpp
  memory/memory_region.cpp
  context/context.cpp
  topology/gpu.cpp
  topology/net.cpp
  topology/node.cpp
  topology/pci.cpp
  topology/system.cpp)

target_include_directories(mori_application_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_application_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
  mori_application_core
  MPI::MPI_CXX
  ibverbs
  hip::host
  hip::device
  mlx5
  rocm_smi64
  pci
  mori_logging
  hsa-runtime64
  hsakmt::hsakmt)

if(USE_BNXT)
  target_link_libraries(mori_application_core ${BNXT_RE_LIB})
endif()

if(USE_IONIC)
  target_link_libraries(mori_application_core ${IONIC_LIB})
endif()

# ---- Torch-specific bootstrap (only when Torch enabled) ----
if(MORI_ENABLE_TORCH)
  execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  cmake_path(SET TORCH_CMAKE_DIR NORMALIZE "${TORCH_DIR}/Torch")
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_DIR})
  message(STATUS "Found LibTorch CMake Path: ${CMAKE_PREFIX_PATH}")

  find_package(Torch REQUIRED)

  add_library(mori_application_torch SHARED bootstrap/torch_bootstrap.cpp)
  target_include_directories(mori_application_torch PUBLIC
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})
  target_link_libraries(mori_application_torch
    mori_application_core c10 torch torch_cpu c10_hip torch_hip)
endif()

# ---- Backward-compatible combined target ----
# Existing code that does target_link_libraries(... mori_application ...) keeps working.
add_library(mori_application INTERFACE)
target_link_libraries(mori_application INTERFACE mori_application_core)
if(MORI_ENABLE_TORCH)
  target_link_libraries(mori_application INTERFACE mori_application_torch)
endif()
