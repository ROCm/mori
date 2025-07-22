add_library(
    mori_io SHARED
    engine.cpp
    protocol.cpp
)

target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(mori_io PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/msgpack-c/include)
target_link_libraries(mori_io mori_application ibverbs hip::host hip::device)
target_compile_definitions(mori_io PUBLIC MSGPACK_NO_BOOST)
