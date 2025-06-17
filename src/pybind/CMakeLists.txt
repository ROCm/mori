find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)

add_library(
    mori_pybinds SHARED
    mori.cpp
    pybind.cpp
)

target_include_directories(mori_pybinds PUBLIC ${PYTHON_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_directories(mori_pybinds PUBLIC ${TORCH_INSTALL_PREFIX}/lib)
target_link_libraries(mori_pybinds mori_ops ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES} torch_python hip::host hip::device)