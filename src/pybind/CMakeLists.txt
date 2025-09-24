option(MORI_WITH_TORCH "Build pybinds with libtorch support" ON)

find_package(PythonLibs REQUIRED)

if(MORI_WITH_TORCH)
  find_package(Torch REQUIRED)
endif()

add_library(mori_pybinds SHARED mori.cpp pybind.cpp)

target_include_directories(mori_pybinds PUBLIC ${PYTHON_INCLUDE_DIRS})
if(MORI_WITH_TORCH)
  target_include_directories(mori_pybinds PUBLIC ${TORCH_INCLUDE_DIRS})
endif()

if(MORI_WITH_TORCH)
  target_link_directories(mori_pybinds PUBLIC ${TORCH_INSTALL_PREFIX}/lib)
endif()

target_link_libraries(mori_pybinds mori_ops mori_io hip::host hip::device)

if(MORI_WITH_TORCH)
  target_link_libraries(mori_pybinds ${TORCH_LIBRARIES} torch_python)
  target_compile_definitions(mori_pybinds PRIVATE MORI_WITH_TORCH=1)
endif()

# For python packages to find dependent libraries
if(MORI_WITH_TORCH)
  set_target_properties(
    mori_pybinds
    PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
               INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
               BUILD_WITH_INSTALL_RPATH TRUE)
else()
  set_target_properties(mori_pybinds PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE)
endif()
