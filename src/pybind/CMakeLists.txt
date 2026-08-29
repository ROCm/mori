# You can set PYTHON_EXECUTABLE via command line:
# -DPYTHON_EXECUTABLE=/path/to/python
if(NOT DEFINED PYTHON_EXECUTABLE)
  # Try to use the same Python that built PyTorch
  execute_process(
    COMMAND python3 -c "import sys; print(sys.executable)"
    OUTPUT_VARIABLE DETECTED_PYTHON
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(DETECTED_PYTHON)
    set(PYTHON_EXECUTABLE ${DETECTED_PYTHON})
    message(STATUS "Auto-detected Python: ${PYTHON_EXECUTABLE}")
  else()
    set(PYTHON_EXECUTABLE "/usr/local/bin/python3.10")
    message(STATUS "Using fallback Python: ${PYTHON_EXECUTABLE}")
  endif()
else()
  message(STATUS "Using manually specified Python: ${PYTHON_EXECUTABLE}")
endif()

set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
find_package(Python REQUIRED COMPONENTS Interpreter Development)

set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
set(PYTHON_LIBRARIES ${Python_LIBRARIES})
message(STATUS "Python include dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${PYTHON_LIBRARIES}")

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_result)
if(NOT _pybind11_result EQUAL 0)
  message(FATAL_ERROR "Failed to find pybind11 include dir")
endif()
message(STATUS "pybind11 include dir: ${PYBIND11_INCLUDE_DIR}")

set(MORI_PYBIND_SOURCES mori.cpp pybind_ops.cpp pybind_shmem.cpp pybind_io.cpp)
if(BUILD_COLLECTIVE)
  list(APPEND MORI_PYBIND_SOURCES pybind_ccl.cpp)
endif()
if(BUILD_UMBP)
  list(APPEND MORI_PYBIND_SOURCES pybind_umbp.cpp)
endif()
add_library(mori_pybinds SHARED ${MORI_PYBIND_SOURCES})

if(ENABLE_PROFILER)
  target_sources(mori_pybinds PRIVATE ${PROFILER_OUTPUT_PYBIND})
  add_dependencies(mori_pybinds generate_profiler_bindings)
endif()

target_include_directories(
  mori_pybinds PUBLIC ${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR}
                      ${CMAKE_BINARY_DIR}/generated/include)
target_link_directories(mori_pybinds PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(mori_pybinds mori_ops mori_shmem mori_io hip::host)
if(BUILD_COLLECTIVE)
  target_link_libraries(mori_pybinds mori_collective)
  target_compile_definitions(mori_pybinds PRIVATE MORI_BUILD_COLLECTIVE)
endif()

if(BUILD_UMBP)
  target_link_libraries(mori_pybinds umbp_core umbp_common)
  target_include_directories(mori_pybinds
                             PRIVATE ${CMAKE_SOURCE_DIR}/src/umbp/include)
  target_compile_definitions(mori_pybinds PRIVATE MORI_BUILD_UMBP)
endif()

if(BUILD_XLA_FFI_OPS)
  target_sources(mori_pybinds PRIVATE pybind_xla_ffi_ops.cpp)

  if(NOT XLA_FFI_INCLUDE_DIR AND DEFINED ENV{XLA_FFI_INCLUDE_DIR})
    set(XLA_FFI_INCLUDE_DIR "$ENV{XLA_FFI_INCLUDE_DIR}")
  endif()

  if(NOT XLA_FFI_INCLUDE_DIR)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c
              "from jax import ffi; print(ffi.include_dir())"
      OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _jax_ffi_result)
    if(NOT _jax_ffi_result EQUAL 0 OR NOT XLA_FFI_INCLUDE_DIR)
      message(
        FATAL_ERROR
          "BUILD_XLA_FFI_OPS=ON: could not resolve XLA FFI headers. Install jax "
          "(providing jax.ffi.include_dir()) or set -DXLA_FFI_INCLUDE_DIR=<path>.")
    endif()
  endif()

  message(STATUS "XLA FFI include dir: ${XLA_FFI_INCLUDE_DIR}")
  if(NOT EXISTS "${XLA_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR "XLA_FFI_INCLUDE_DIR='${XLA_FFI_INCLUDE_DIR}' does not contain xla/ffi/api/ffi.h")
  endif()
  target_include_directories(mori_pybinds PRIVATE ${XLA_FFI_INCLUDE_DIR})
endif()

# For python packages to find dependent libraries
set_target_properties(
  mori_pybinds
  PROPERTIES BUILD_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             INSTALL_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
