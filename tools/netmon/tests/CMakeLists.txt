add_executable(netmon_tests test_main.cc test_parsers.cc test_util.cc
                            test_buffers.cc test_recorder.cc test_push.cc)
target_link_libraries(netmon_tests PRIVATE netmon_core)
target_compile_options(netmon_tests PRIVATE -Wall -Wextra)
target_compile_definitions(
  netmon_tests PRIVATE NETMON_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME netmon_tests COMMAND netmon_tests)

add_executable(netmon_live_replay live_replay.cc)
target_link_libraries(netmon_live_replay PRIVATE netmon_core)
target_compile_options(netmon_live_replay PRIVATE -Wall -Wextra)
set_tests_properties(netmon_tests PROPERTIES TIMEOUT 300)

# Report regression tests: only when a Python 3 with numpy is available (plotly
# is not needed).
find_package(
  Python3
  COMPONENTS Interpreter
  QUIET)
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy"
    RESULT_VARIABLE NETMON_NUMPY_MISSING
    OUTPUT_QUIET ERROR_QUIET)
  if(NOT NETMON_NUMPY_MISSING)
    add_test(
      NAME netmon_report_tests
      COMMAND ${CMAKE_COMMAND} -E env NETMON_LIVE_REPLAY=$<TARGET_FILE:netmon_live_replay>
              ${Python3_EXECUTABLE} -m unittest -v test_report test_import test_live
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(STATUS "ainic-netmon: numpy not found, report tests skipped")
  endif()
endif()
