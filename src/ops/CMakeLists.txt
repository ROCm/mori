add_library(
    mori_ops
    dispatch_combine/dispatch_combine.cpp
)
target_link_libraries(mori_ops mori_application mori_shmem hip::host hip::device)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/include)