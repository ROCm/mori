add_executable(netmon_tests test_main.cc test_parsers.cc test_util.cc
                            test_buffers.cc test_recorder.cc)
target_link_libraries(netmon_tests PRIVATE netmon_core)
target_compile_options(netmon_tests PRIVATE -Wall -Wextra)
target_compile_definitions(
  netmon_tests PRIVATE NETMON_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME netmon_tests COMMAND netmon_tests)
set_tests_properties(netmon_tests PROPERTIES TIMEOUT 300)
