find_package(MPI REQUIRED)

option(BUILD_SHMEM_DEVICE_WRAPPER "Build SHMEM device API wrapper" OFF)

set(MORI_SHMEM_SOURCES init.cpp memory.cpp)
if(BUILD_SHMEM_DEVICE_WRAPPER)
  list(APPEND MORI_SHMEM_SOURCES shmem_device_api_wrapper.cpp)
  message(STATUS "BUILD_SHMEM_DEVICE_WRAPPER enabled for mori_shmem")
endif()

# These files contain __device__ code and must be compiled with hipcc
set_source_files_properties(${MORI_SHMEM_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(mori_shmem ${MORI_SHMEM_SOURCES})

target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
  mori_shmem
  PUBLIC mori_application mori_logging MPI::MPI_CXX ibverbs hip::host mlx5)

# Device code is self-contained in this shared library
target_compile_options(mori_shmem PRIVATE "-fgpu-rdc" "-mcode-object-version=5")
target_link_options(mori_shmem PRIVATE "-fgpu-rdc" "-mcode-object-version=5")

# Device NIC provider for IBGDA dispatch (single provider, detected at top level)
if(MORI_DEVICE_NIC_DEFINE)
  target_compile_definitions(mori_shmem PRIVATE ${MORI_DEVICE_NIC_DEFINE})
endif()

option(SAVE_TEMPS "Save intermediate compilation files" ON)
if(SAVE_TEMPS)
  target_compile_options(mori_shmem PRIVATE "-save-temps=obj")
  message(STATUS "SAVE_TEMPS enabled for mori_shmem - intermediate files will be saved")
  message(STATUS "BC files will be in: ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/mori_shmem.dir/")
endif()
