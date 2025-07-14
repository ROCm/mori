add_library(
    mori_io SHARED
    engine.cpp
)

target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mori_io hip::host hip::device)
