set(MORI_OPS_SOURCES dispatch_combine/dispatch_combine.cpp
                     dispatch_combine/launch.cpp)

add_library(mori_ops SHARED ${MORI_OPS_SOURCES})

target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mori_ops PUBLIC mori_application mori_shmem mori_logging
                                      hip::host dl)

set_target_properties(
  mori_ops
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)

if(ENABLE_PROFILER)
  target_link_libraries(mori_ops PUBLIC mori_profiler_interface)
endif()

# ---------------------------------------------------------------------------
# BUILD_OPS_DEVICE: AOT compile EP kernels to .hsaco launch.cpp is always
# compiled (host-only, pure CXX). This option only controls whether .hsaco
# device modules are built. GPU_TARGETS and MORI_DEVICE_NIC_DEFINE are set by
# the top-level CMakeLists.
# ---------------------------------------------------------------------------
if(BUILD_OPS_DEVICE)
  find_program(HIPCC_EXECUTABLE hipcc PATHS /opt/rocm/bin ENV ROCM_PATH)
  if(NOT HIPCC_EXECUTABLE)
    message(
      FATAL_ERROR
        "BUILD_OPS_DEVICE=ON requires hipcc. Set ROCM_PATH or install ROCm.")
  endif()

  # NIC and feature defines (from top-level CMake)
  set(_nic_defs "")
  if(MORI_DEVICE_NIC_DEFINE)
    set(_nic_defs "-D${MORI_DEVICE_NIC_DEFINE}")
  endif()

  set(_extra_defs "")
  if(ENABLE_PROFILER)
    list(APPEND _extra_defs "-DENABLE_PROFILER")
  endif()
  if(ENABLE_STANDARD_MOE_ADAPT)
    list(APPEND _extra_defs "-DENABLE_STANDARD_MOE_ADAPT")
  endif()

  # MPI include dirs for hipcc genco (headers include mpi.h)
  find_package(MPI QUIET)
  set(_mpi_includes "")
  if(MPI_CXX_INCLUDE_DIRS)
    foreach(_d ${MPI_CXX_INCLUDE_DIRS})
      list(APPEND _mpi_includes "-I${_d}")
    endforeach()
  endif()

  # Kernel sources
  set(_kernel_sources ep_intranode.hip ep_internode.hip ep_internode_v1.hip
                      ep_internode_v1ll.hip ep_async_ll.hip)

  # Output to build/lib/<arch>_<nic>/ (matches JIT cache layout)
  set(_arch_nic "${GPU_TARGETS}_${MORI_DEVICE_NIC}")
  set(_hsaco_dir "${CMAKE_BINARY_DIR}/lib/${_arch_nic}")
  file(MAKE_DIRECTORY ${_hsaco_dir})

  set(_all_hsaco_outputs "")
  foreach(_ksrc ${_kernel_sources})
    get_filename_component(_kname ${_ksrc} NAME_WE)
    set(_hsaco_output "${_hsaco_dir}/${_kname}.hsaco")
    set(_ksrc_path "${CMAKE_SOURCE_DIR}/src/ops/kernels/${_ksrc}")

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
      set(FLAGS "-O2")
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(FLAGS "-g3" "-ggdb" "-O2")
    endif()
    add_custom_command(
      OUTPUT ${_hsaco_output}
      COMMAND
        ${HIPCC_EXECUTABLE} --genco "--offload-arch=${GPU_TARGETS}" -std=c++17
        ${FLAGS} -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_WARP_SYNC_BUILTINS ${_nic_defs}
        ${_extra_defs} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}
        ${_mpi_includes} ${_ksrc_path} -o ${_hsaco_output}
      DEPENDS ${_ksrc_path}
      COMMENT "AOT compiling ${_ksrc} -> ${_arch_nic}/${_kname}.hsaco"
      VERBATIM)
    list(APPEND _all_hsaco_outputs ${_hsaco_output})
  endforeach()

  add_custom_target(mori_ops_kernels ALL DEPENDS ${_all_hsaco_outputs})
  add_dependencies(mori_ops mori_ops_kernels)

  install(FILES ${_all_hsaco_outputs} DESTINATION "lib/${_arch_nic}")

  message(
    STATUS "BUILD_OPS_DEVICE: will AOT compile kernels -> lib/${_arch_nic}/")
endif()
