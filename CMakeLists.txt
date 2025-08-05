cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_ROCM "Whether to use rocm" ON)
option(BUILD_EXAMPLES "Whether to build examples" ON)
option(BUILD_APPLICATION "Whether to build application library" ON)
option(BUILD_SHMEM "Whether to build shmem library" ON)
option(BUILD_OPS "Whether to build mori operation kernels" ON)
option(BUILD_IO "Whether to build mori io library" ON)
option(BUILD_PYBINDS "Whether to build mori python bindings" ON)
option(BUILD_TESTS "Whether to build mori CPP tests" ON)


if (USE_ROCM)
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    project(
        mori
        LANGUAGES HIP CXX C
    )
    # set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
    find_package(hip REQUIRED)
else()
    set(CUDA_HOME /usr/local/cuda)
    message(STATUS "CUDA_HOME: ${CUDA_HOME}")
    set(CUDAToolkit_Root ${CUDA_HOME} CACHE PATH "Root of Cuda Toolkit." FORCE)
    set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc" CACHE PATH "Root of Cuda Toolkit." FORCE)
    set(CMAKE_CUDA_FLAGS "--generate-code=arch=compute_70,code=[sm_70] --generate-code=arch=compute_80,code=[sm_80]")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

    project(
        mori
        LANGUAGES CUDA CXX C
    )
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr")
endif()
find_package(glog REQUIRED)


if (BUILD_EXAMPLES)
    SET(BUILD_APPLICATION ON)
    add_subdirectory(examples)
endif()

if (BUILD_APPLICATION)
    add_subdirectory(src/application)
endif()

if (BUILD_SHMEM)
    add_subdirectory(src/shmem)
endif()

if (BUILD_OPS)
    add_subdirectory(src/ops)
endif()

if (BUILD_IO)
    add_subdirectory(src/io)
endif()

if (BUILD_PYBINDS)
    add_subdirectory(src/pybind)
endif()

if (BUILD_TESTS)
    add_subdirectory(tests/cpp)
endif()
