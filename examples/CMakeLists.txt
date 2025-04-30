

add_executable(
    custom_allreduce
    custom_allreduce/custom_allreduce.cpp
)

target_link_libraries(custom_allreduce mori_application hip::host hip::device)