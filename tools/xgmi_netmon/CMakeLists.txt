cmake_minimum_required(VERSION 3.16)
project(
  xgmi_netmon
  VERSION 0.1.0
  LANGUAGES CXX)

option(XGMI_NETMON_BUILD_TESTS "Build xgmi-netmon unit tests" ON)
option(XGMI_NETMON_TSAN "Build with ThreadSanitizer" OFF)

find_package(Threads REQUIRED)

add_library(
  xgmi_netmon_core STATIC
  src/gpu_metrics.cc
  src/http.cc
  src/import.cc
  src/rates.cc
  src/recorder.cc
  src/recording.cc
  src/topology.cc
  src/vm.cc)
target_include_directories(xgmi_netmon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(xgmi_netmon_core PUBLIC cxx_std_17)
set_target_properties(xgmi_netmon_core PROPERTIES CXX_EXTENSIONS OFF)
target_compile_definitions(xgmi_netmon_core
                           PUBLIC XGMI_NETMON_VERSION="${PROJECT_VERSION}")
target_compile_options(xgmi_netmon_core PRIVATE -Wall -Wextra -Werror)
target_link_libraries(xgmi_netmon_core PUBLIC Threads::Threads)
if(XGMI_NETMON_TSAN)
  target_compile_options(xgmi_netmon_core PUBLIC -fsanitize=thread -g)
  target_link_options(xgmi_netmon_core PUBLIC -fsanitize=thread)
endif()

add_executable(xgmi-netmon src/main.cc)
target_compile_options(xgmi-netmon PRIVATE -Wall -Wextra -Werror)
target_link_libraries(xgmi-netmon PRIVATE xgmi_netmon_core)
# Only glibc stays dynamic, so the binary can be copied to other hosts.
if(NOT XGMI_NETMON_TSAN)
  target_link_options(xgmi-netmon PRIVATE -static-libstdc++ -static-libgcc)
endif()

if(XGMI_NETMON_BUILD_TESTS)
  enable_testing()
  add_executable(xgmi_netmon_tests tests/test_main.cc)
  target_compile_options(xgmi_netmon_tests PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(xgmi_netmon_tests PRIVATE xgmi_netmon_core)
  target_compile_definitions(
    xgmi_netmon_tests
    PRIVATE XGMI_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
  add_test(NAME xgmi_netmon_tests COMMAND xgmi_netmon_tests)
endif()
