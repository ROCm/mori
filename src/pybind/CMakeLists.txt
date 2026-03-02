# You can set PYTHON_EXECUTABLE via command line: -DPYTHON_EXECUTABLE=/path/to/python
if(NOT DEFINED PYTHON_EXECUTABLE)
  # Try to use the same Python that built PyTorch
  execute_process(
    COMMAND python3 -c "import sys; print(sys.executable)"
    OUTPUT_VARIABLE DETECTED_PYTHON
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
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
  COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_result)
if(NOT _pybind11_result EQUAL 0)
  message(FATAL_ERROR "Failed to find pybind11 include dir")
endif()
message(STATUS "pybind11 include dir: ${PYBIND11_INCLUDE_DIR}")

add_library(mori_pybinds SHARED mori.cpp pybind_ops.cpp pybind_shmem.cpp pybind_io.cpp)

if(ENABLE_PROFILER)
  target_sources(mori_pybinds PRIVATE ${PROFILER_OUTPUT_PYBIND})
  add_dependencies(mori_pybinds generate_profiler_bindings)
endif()

target_include_directories(
  mori_pybinds PUBLIC ${PYTHON_INCLUDE_DIRS}
                      ${PYBIND11_INCLUDE_DIR}
                      ${CMAKE_BINARY_DIR}/generated/include)
target_link_libraries(
  mori_pybinds
  mori_ops
  mori_io
  hip::host
  hip::device)

# For python packages to find dependent libraries
set_target_properties(
  mori_pybinds
  PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
             INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
