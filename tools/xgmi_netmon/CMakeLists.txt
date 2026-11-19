cmake_minimum_required(VERSION 3.16)
project(
  xgmi_netmon
  VERSION 0.1.0
  LANGUAGES CXX)

# Standalone (cmake -S tools/xgmi_netmon) or a subdirectory of mori, which builds it so the wheel
# can ship it; the options then stay private to these targets and the tests are left out.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(XGMI_NETMON_STANDALONE ON)
else()
  set(XGMI_NETMON_STANDALONE OFF)
endif()

option(XGMI_NETMON_BUILD_TESTS "Build xgmi-netmon unit tests"
       ${XGMI_NETMON_STANDALONE})
option(XGMI_NETMON_WERROR "Treat warnings as errors" ${XGMI_NETMON_STANDALONE})
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
set_target_properties(xgmi_netmon_core PROPERTIES CXX_EXTENSIONS OFF
                                                  POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(xgmi_netmon_core
                           PUBLIC XGMI_NETMON_VERSION="${PROJECT_VERSION}")
target_compile_options(xgmi_netmon_core PRIVATE -Wall -Wextra)
if(XGMI_NETMON_WERROR)
  target_compile_options(xgmi_netmon_core PRIVATE -Werror)
endif()
target_link_libraries(xgmi_netmon_core PUBLIC Threads::Threads)
if(XGMI_NETMON_TSAN)
  target_compile_options(xgmi_netmon_core PUBLIC -fsanitize=thread -g)
  target_link_options(xgmi_netmon_core PUBLIC -fsanitize=thread)
endif()

add_executable(xgmi-netmon src/main.cc)
target_compile_options(xgmi-netmon PRIVATE -Wall -Wextra)
if(XGMI_NETMON_WERROR)
  target_compile_options(xgmi-netmon PRIVATE -Werror)
endif()
target_link_libraries(xgmi-netmon PRIVATE xgmi_netmon_core)
# Only glibc stays dynamic, so the binary can be copied to other hosts.
if(NOT XGMI_NETMON_TSAN)
  target_link_options(xgmi-netmon PRIVATE -static-libstdc++ -static-libgcc)
endif()
# Never inherit a parent project's RPATH (e.g. ROCm).
set_target_properties(xgmi-netmon PROPERTIES SKIP_BUILD_RPATH ON INSTALL_RPATH
                                                                 "")
install(TARGETS xgmi-netmon RUNTIME DESTINATION bin)

if(XGMI_NETMON_BUILD_TESTS)
  enable_testing()
  add_executable(xgmi_netmon_tests tests/test_main.cc)
  target_compile_options(xgmi_netmon_tests PRIVATE -Wall -Wextra)
  if(XGMI_NETMON_WERROR)
    target_compile_options(xgmi_netmon_tests PRIVATE -Werror)
  endif()
  target_link_libraries(xgmi_netmon_tests PRIVATE xgmi_netmon_core)
  target_compile_definitions(
    xgmi_netmon_tests
    PRIVATE XGMI_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
  add_test(NAME xgmi_netmon_tests COMMAND xgmi_netmon_tests)
endif()
