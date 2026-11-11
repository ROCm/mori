# ainic-netmon: millisecond-level counter collector for AMD AINIC (ionic).
#
# Standalone project: depends only on libc and Linux kernel headers (no mori, no
# ROCm). Build commands are in README.md. When added as a subdirectory of a
# larger project, all options stay private to these targets.
cmake_minimum_required(VERSION 3.16)
project(
  ainic_netmon
  VERSION 0.1.0
  LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(NETMON_STANDALONE ON)
else()
  set(NETMON_STANDALONE OFF)
endif()

option(NETMON_BUILD_TESTS "Build ainic-netmon unit tests" ${NETMON_STANDALONE})
option(NETMON_WERROR "Treat warnings as errors" ${NETMON_STANDALONE})
option(NETMON_TSAN "Build with ThreadSanitizer (tests)" OFF)

find_package(Threads REQUIRED)

set(NETMON_SOURCES
    src/cli.cc
    src/discover.cc
    src/ethtool_stats.cc
    src/guarded_buffer.cc
    src/ionic_ipc.cc
    src/meta_json.cc
    src/privilege.cc
    src/rdma_netlink.cc
    src/recorder.cc
    src/sampler.cc
    src/util.cc
    src/writer.cc)

# Core library shared by the binary and the tests.
add_library(netmon_core STATIC ${NETMON_SOURCES})
target_include_directories(netmon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(netmon_core PUBLIC cxx_std_17)
set_target_properties(netmon_core PROPERTIES CXX_EXTENSIONS OFF
                                             POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(netmon_core
                           PUBLIC NETMON_VERSION="${PROJECT_VERSION}")
target_compile_options(netmon_core PRIVATE -Wall -Wextra)
if(NETMON_WERROR)
  target_compile_options(netmon_core PRIVATE -Werror)
endif()
if(NETMON_TSAN)
  target_compile_options(netmon_core PUBLIC -fsanitize=thread -g -O1)
  target_link_options(netmon_core PUBLIC -fsanitize=thread)
endif()
target_link_libraries(netmon_core PUBLIC Threads::Threads)

add_executable(ainic-netmon src/main.cc)
target_link_libraries(ainic-netmon PRIVATE netmon_core)
target_compile_options(ainic-netmon PRIVATE -Wall -Wextra)
if(NETMON_WERROR)
  target_compile_options(ainic-netmon PRIVATE -Werror)
endif()
if(NOT NETMON_TSAN)
  # Only glibc stays dynamic, so the binary copies between distributions and
  # containers.
  target_link_options(ainic-netmon PRIVATE -static-libstdc++ -static-libgcc)
endif()
# Never inherit a parent project's RPATH (e.g. ROCm).
set_target_properties(ainic-netmon PROPERTIES SKIP_BUILD_RPATH ON INSTALL_RPATH
                                                                  "")
install(TARGETS ainic-netmon RUNTIME DESTINATION bin)

if(NETMON_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
