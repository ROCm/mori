if(WITH_MPI)
  find_package(MPI)
  if(NOT MPI_FOUND)
    message(
      FATAL_ERROR
        "MPI not found, but WITH_MPI=ON (enabled because BUILD_EXAMPLES=ON).\n"
        "Please install MPI (e.g. 'apt install libopenmpi-dev') or disable examples with "
        "'-DBUILD_EXAMPLES=OFF' / 'BUILD_EXAMPLES=OFF pip install .'")
  endif()
endif()
find_package(hsa-runtime64 REQUIRED)
find_package(hsakmt REQUIRED)
# find_library(IONIC_LIBRARY NAMES ionic HINTS /lib/x86_64-linux-gnu REQUIRED )
option(BUILD_TORCH_BOOTSTRAP
       "Build Torch process group bootstrap (requires PyTorch)" ON)

if(BUILD_TORCH_BOOTSTRAP)
  execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_RESULT)
  if(TORCH_RESULT EQUAL 0 AND TORCH_DIR)
    cmake_path(SET TORCH_CMAKE_DIR NORMALIZE "${TORCH_DIR}/Torch")
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_DIR})
    find_package(Torch QUIET)
  endif()
  if(NOT Torch_FOUND)
    message(WARNING "PyTorch not found — disabling Torch bootstrap. "
                    "MPI and socket bootstrap are still available.")
    set(BUILD_TORCH_BOOTSTRAP OFF)
  else()
    message(STATUS "Found LibTorch CMake Path: ${CMAKE_PREFIX_PATH}")
  endif()
endif()

set(MORI_APP_SOURCES bootstrap/socket_bootstrap.cpp
                     bootstrap/local_bootstrap.cpp)

if(WITH_MPI)
  list(PREPEND MORI_APP_SOURCES bootstrap/mpi_bootstrap.cpp)
endif()

if(BUILD_TORCH_BOOTSTRAP)
  list(APPEND MORI_APP_SOURCES bootstrap/torch_bootstrap.cpp)
endif()

list(
  APPEND
  MORI_APP_SOURCES
  transport/rdma/rdma.cpp
  transport/rdma/providers/dv_loader.cpp
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

set_source_files_properties(${MORI_APP_SOURCES} PROPERTIES LANGUAGE CXX)
add_library(mori_application SHARED ${MORI_APP_SOURCES})

target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR})
find_library(ROCM_SMI_LIB rocm_smi64 HINTS /opt/rocm/lib REQUIRED)
find_library(HSA_RUNTIME_LIB hsa-runtime64 HINTS /opt/rocm/lib REQUIRED)
target_link_libraries(
  mori_application
  PUBLIC ibverbs
         hip::host
         dl
         ${ROCM_SMI_LIB}
         pci
         mori_logging
         ${HSA_RUNTIME_LIB}
         hsakmt::hsakmt)

if(WITH_MPI)
  target_compile_definitions(mori_application PUBLIC MORI_WITH_MPI)
  target_link_libraries(mori_application PUBLIC MPI::MPI_CXX)
endif()

if(BUILD_TORCH_BOOTSTRAP)
  target_compile_definitions(mori_application PUBLIC MORI_HAS_TORCH)
  target_include_directories(mori_application PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(mori_application PRIVATE c10 torch torch_cpu c10_hip
                                                 torch_hip)
  set_target_properties(
    mori_application
    PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
               INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
               BUILD_WITH_INSTALL_RPATH TRUE)
else()
  set_target_properties(
    mori_application
    PROPERTIES BUILD_RPATH "$ORIGIN"
               INSTALL_RPATH "$ORIGIN"
               BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Export only mori symbols (to avoid leaking spdlog symbols)
target_link_options(mori_application PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/mori_exports.lds"
)