add_library(
    mori_ops
    dispatch_combine/dispatch_combine.cpp
)
target_link_libraries(mori_ops mori_application mori_shmem hip::host hip::device)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/include)

# add_library(
#     mori_dispatch_combine SHARED
#     moe/pybinds/dispatch_combine.cpp
#     moe/pybinds/bindings.cpp
#     moe/dispatch_combine_kernels/dispatch_combine.cpp
# )
# target_link_libraries(mori_dispatch_combine mori_application mori_shmem hip::host hip::device)
# target_include_directories(mori_dispatch_combine PUBLIC ${CMAKE_SOURCE_DIR}/examples/moe)
