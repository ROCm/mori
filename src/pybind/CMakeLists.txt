find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)

add_library(mori_pybinds SHARED mori.cpp pybind.cpp)

if(ENABLE_PROFILER)
  target_sources(mori_pybinds PRIVATE ${PROFILER_OUTPUT_PYBIND})
  add_dependencies(mori_pybinds generate_profiler_bindings)
endif()

target_include_directories(
  mori_pybinds PUBLIC ${PYTHON_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
                      ${CMAKE_BINARY_DIR}/generated/include)
target_link_directories(mori_pybinds PUBLIC ${TORCH_INSTALL_PREFIX}/lib)
target_link_libraries(
  mori_pybinds
  mori_ops
  mori_io
  ${TORCH_LIBRARIES}
  torch_python
  hip::host
  hip::device)

# For python packages to find dependent libraries
set_target_properties(
  mori_pybinds
  PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
             INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
