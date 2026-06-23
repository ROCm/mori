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

# ROCm 7.14's hsakmt cmake config declares a link dependency on the imported
# target `numa::numa` but never provides it (no FindNUMA / find_package(NUMA)).
# Without it, find_package(hsakmt) fails at generate time: "The link interface
# of target hsakmt::hsakmt contains: numa::numa but the target was not found."
# Define it ourselves from the system libnuma before pulling in hsakmt.
if(NOT TARGET numa::numa)
  find_library(NUMA_LIBRARY NAMES numa)
  find_path(NUMA_INCLUDE_DIR NAMES numa.h)
  if(NUMA_LIBRARY)
    add_library(numa::numa UNKNOWN IMPORTED)
    set_target_properties(numa::numa PROPERTIES IMPORTED_LOCATION
                                                "${NUMA_LIBRARY}")
    if(NUMA_INCLUDE_DIR)
      set_target_properties(numa::numa PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                  "${NUMA_INCLUDE_DIR}")
    endif()
  else()
    message(FATAL_ERROR "libnuma not found but required by hsakmt::hsakmt. "
                        "Install it (e.g. 'apt install libnuma-dev').")
  endif()
endif()

find_package(hsakmt REQUIRED)
# find_library(IONIC_LIBRARY NAMES ionic HINTS /lib/x86_64-linux-gnu REQUIRED )

set(MORI_APP_SOURCES bootstrap/socket_bootstrap.cpp
                     bootstrap/local_bootstrap.cpp)

if(WITH_MPI)
  list(PREPEND MORI_APP_SOURCES bootstrap/mpi_bootstrap.cpp)
endif()

list(
  APPEND
  MORI_APP_SOURCES
  transport/rdma/rdma.cpp
  transport/rdma/providers/dv_loader.cpp
  transport/rdma/providers/mlx5/mlx5.cpp
  transport/rdma/providers/mlx5/mlx5_abi_parity.cpp
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

# libibverbs is loaded at runtime via dlopen (see ibv_shim.cpp), not linked. The
# shim is built as an object library with hidden visibility so every mori shared
# object that touches the verbs API embeds its own private copy and never
# interposes a real libibverbs that may also be present in the process.
add_library(mori_ibv_shim OBJECT transport/rdma/providers/ibverbs/ibv_shim.cpp)
set_source_files_properties(transport/rdma/providers/ibverbs/ibv_shim.cpp
                            PROPERTIES LANGUAGE CXX)
target_include_directories(mori_ibv_shim PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mori_ibv_shim PRIVATE mori_logging)
target_link_libraries(mori_ibv_shim PRIVATE dl)
target_compile_options(mori_ibv_shim PRIVATE -fvisibility=hidden
                                             -fvisibility-inlines-hidden)
set_target_properties(mori_ibv_shim PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_source_files_properties(${MORI_APP_SOURCES} PROPERTIES LANGUAGE CXX)
add_library(mori_application SHARED ${MORI_APP_SOURCES})

target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_application PUBLIC ${CMAKE_SOURCE_DIR})
find_library(HSA_RUNTIME_LIB hsa-runtime64 HINTS /opt/rocm/lib REQUIRED)
target_link_libraries(mori_application PUBLIC hip::host dl pci mori_logging
                                              ${HSA_RUNTIME_LIB} hsakmt::hsakmt)
# Embed the libibverbs dlopen shim (object library, hidden visibility).
target_link_libraries(mori_application PRIVATE mori_ibv_shim)

if(WITH_MPI)
  target_compile_definitions(mori_application PUBLIC MORI_WITH_MPI)
  target_link_libraries(mori_application PUBLIC MPI::MPI_CXX)
endif()

set_target_properties(
  mori_application
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
