if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

option(BUILD_SHMEM_DEVICE_WRAPPER "Build SHMEM device API wrapper" OFF)

if(BUILD_SHMEM_DEVICE_WRAPPER)
  add_library(mori_shmem init.cpp memory.cpp shmem_device_api_wrapper.cpp)
else()
  add_library(mori_shmem init.cpp memory.cpp)
endif()

target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR} 
                                             ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(
  mori_shmem
  mori_application
  mori_logging
  ibverbs
  hip::host
  hip::device
  mlx5)

if(ENABLE_MPI)
  target_link_libraries(mori_shmem MPI::MPI_CXX)
endif()

target_compile_options(mori_shmem PUBLIC "-fgpu-rdc")
target_link_options(mori_shmem PUBLIC "-fgpu-rdc")
if(FORCE_CODE_OBJECT_VERSION_5) 
  # Use code object version 5 for compatibility with Triton
  target_compile_options(mori_shmem PUBLIC "-mcode-object-version=5")
  target_link_options(mori_shmem PUBLIC "-mcode-object-version=5")
endif()

if(NOT ENABLE_TORCH)
  # This seems to be required for compilation for JAX
  set_property(TARGET mori_shmem PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

option(SAVE_TEMPS "Save intermediate compilation files" ON)
if(SAVE_TEMPS)
  target_compile_options(mori_shmem PUBLIC "-save-temps=obj")
  message(STATUS "SAVE_TEMPS enabled for mori_shmem - intermediate files will be saved")
  message(STATUS "BC files will be in: ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/mori_shmem.dir/")
endif()
