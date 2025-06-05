add_library(
    mori_pybinds SHARED
    ops.cpp
    pybind.cpp
)
target_link_libraries(mori_pybinds mori_ops hip::host hip::device)