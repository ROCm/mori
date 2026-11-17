# mori-config.cmake — CMake package config for mori.
#
# After find_package(mori), the following imported targets are available:
# mori::shmem         — shmem host library (link for host code)
# mori::application   — application/transport library mori::ops           — EP
# dispatch/combine host library mori::io            — IO engine library
# mori::cco           — CCO host library (cco.hpp C++ API, cco_c.h C ABI)
#
# And the following functions: mori_detect_device_config()  — detect GPU arch +
# NIC on the current machine mori_add_device_target(tgt)  — apply detected
# config to a HIP target
#
# Variables set by this config: MORI_INCLUDE_DIR    — path to mori public
# headers MORI_LIB_DIR        — path to mori shared libraries
#
# Example: find_package(mori REQUIRED) mori_detect_device_config()
#
# add_executable(my_app my_kernel.hip) set_source_files_properties(my_kernel.hip
# PROPERTIES LANGUAGE HIP) target_link_libraries(my_app mori::shmem hip::device)
# mori_add_device_target(my_app)

cmake_minimum_required(VERSION 3.19)

get_filename_component(_mori_cmake_dir "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(_mori_prefix "${_mori_cmake_dir}/../../.." ABSOLUTE)

set(MORI_INCLUDE_DIR
    "${_mori_prefix}/include"
    CACHE PATH "Mori include directory")
set(MORI_LIB_DIR
    "${_mori_prefix}/lib"
    CACHE PATH "Mori library directory")

# Include device detection helpers
include("${_mori_cmake_dir}/MoriDetectDevice.cmake")

# ---------------------------------------------------------------------------
# Imported targets
# ---------------------------------------------------------------------------
macro(_mori_import_library name)
  if(NOT TARGET mori::${name})
    add_library(mori::${name} SHARED IMPORTED)
    set(_lib_path "${MORI_LIB_DIR}/libmori_${name}.so")
    if(EXISTS "${_lib_path}")
      set_target_properties(
        mori::${name}
        PROPERTIES IMPORTED_LOCATION "${_lib_path}"
                   INTERFACE_INCLUDE_DIRECTORIES
                   "${MORI_INCLUDE_DIR};${MORI_INCLUDE_DIR}/..")
    else()
      message(WARNING "mori::${name}: library not found at ${_lib_path}")
    endif()
  endif()
endmacro()

_mori_import_library(application)
_mori_import_library(shmem)
_mori_import_library(ops)
_mori_import_library(io)
# Absent when MORI was built with BUILD_CCO=OFF. libmori_cco.so is the link-time
# name of libmori_cco.so.1; dlopen users want MORI_CCO_LIBRARY_SONAME (cco_c.h).
if(EXISTS "${MORI_LIB_DIR}/libmori_cco.so.1")
  _mori_import_library(cco)
  set_target_properties(mori::cco PROPERTIES IMPORTED_SONAME "libmori_cco.so.1")
  # cco.hpp lays out its SDMA state by BUILD_CCO_SDMA; match the library.
  include("${_mori_cmake_dir}/mori-build-options.cmake" OPTIONAL)
  if(DEFINED MORI_BUILD_CCO_SDMA)
    if(MORI_BUILD_CCO_SDMA)
      set(_mori_cco_sdma 1)
    else()
      set(_mori_cco_sdma 0)
    endif()
    set_property(
      TARGET mori::cco
      APPEND
      PROPERTY INTERFACE_COMPILE_DEFINITIONS BUILD_CCO_SDMA=${_mori_cco_sdma})
    unset(_mori_cco_sdma)
  endif()
endif()
# Absent when MORI was built with BUILD_RAS=OFF.
if(EXISTS "${MORI_LIB_DIR}/libmori_ras.so")
  _mori_import_library(ras)
endif()

# Set inter-library dependencies
if(TARGET mori::ras AND TARGET mori::io)
  set_property(
    TARGET mori::io
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES mori::ras)
endif()
if(TARGET mori::shmem)
  set_property(
    TARGET mori::shmem
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES mori::application)
endif()
if(TARGET mori::ops)
  set_property(
    TARGET mori::ops
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES mori::shmem mori::application)
endif()

# MPI is typically required for shmem initialization
find_package(MPI QUIET)
if(MPI_FOUND AND TARGET mori::shmem)
  set_property(
    TARGET mori::shmem
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES MPI::MPI_CXX)
endif()

unset(_mori_cmake_dir)
unset(_mori_prefix)
