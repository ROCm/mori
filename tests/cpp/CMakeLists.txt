add_executable(
    test_transport_tcp
    application/test_transport_tcp.cpp
)

target_include_directories(test_transport_tcp PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_transport_tcp PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_transport_tcp mori_application)

add_executable(
    test_engine
    io/test_engine.cpp
)

target_include_directories(test_engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_engine PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_engine mori_application mori_io)
