add_executable(
    test_transport_tcp
    application/test_transport_tcp.cpp
)

target_include_directories(test_transport_tcp PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_transport_tcp PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(test_transport_tcp mori_application)
