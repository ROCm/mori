# JIT v2 tests.
add_executable(test_jit_core test_jit_core.cpp)
target_include_directories(test_jit_core PUBLIC ${CMAKE_SOURCE_DIR}
                                                ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_jit_core mori_jit mori_ops_v2 gtest_main)
set_target_properties(test_jit_core PROPERTIES CXX_STANDARD 20
                                               CXX_STANDARD_REQUIRED ON)
add_test(NAME jit_core COMMAND test_jit_core)

# The host/device split guard: cfg headers must stay compilable without hipcc.
add_test(NAME jit_host_device_split
         COMMAND ${CMAKE_SOURCE_DIR}/tools/jit_v2/check_host_device_split.sh
                 ${CMAKE_SOURCE_DIR})
