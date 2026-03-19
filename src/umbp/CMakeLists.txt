# ---------------------------------------------------------------------------
# SPDK auto-detection: try pkg-config, enable if found.
# Set SPDK_PKG_CONFIG_PATH to override search path for non-standard installs.
# Set USE_SPDK=OFF to force-disable even if SPDK is installed.
# ---------------------------------------------------------------------------
set(SPDK_PKG_CONFIG_PATH "" CACHE STRING "Custom pkg-config path for SPDK")
option(USE_SPDK "Enable SPDK support (auto-detected by default)" ON)

set(HAVE_SPDK FALSE)
if(USE_SPDK)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    if(SPDK_PKG_CONFIG_PATH)
      set(ENV{PKG_CONFIG_PATH} "${SPDK_PKG_CONFIG_PATH}:$ENV{PKG_CONFIG_PATH}")
    endif()

    execute_process(
      COMMAND pkg-config --cflags spdk_event spdk_event_bdev spdk_env_dpdk
      OUTPUT_VARIABLE SPDK_CFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE SPDK_CFLAGS_RC
      ERROR_QUIET)
    execute_process(
      COMMAND pkg-config --libs --static spdk_event spdk_event_bdev spdk_env_dpdk
      OUTPUT_VARIABLE SPDK_LDFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE SPDK_LDFLAGS_RC
      ERROR_QUIET)

    if(SPDK_CFLAGS_RC EQUAL 0 AND SPDK_LDFLAGS_RC EQUAL 0)
      set(HAVE_SPDK TRUE)
    endif()
  endif()

  if(HAVE_SPDK)
    message(STATUS "[UMBP] SPDK detected via pkg-config — SPDK support ENABLED")
    message(STATUS "[UMBP] SPDK CFLAGS: ${SPDK_CFLAGS}")
    message(STATUS "[UMBP] SPDK LDFLAGS: ${SPDK_LDFLAGS}")
  else()
    message(STATUS "[UMBP] SPDK not found via pkg-config — SPDK support DISABLED (POSIX only)")
  endif()
endif()

# Core static library
set(UMBP_CORE_SRCS
  block_index/block_index.cpp storage/tier_backend.cpp storage/dram_tier.cpp
  storage/ssd_tier.cpp storage/local_storage_manager.cpp umbp_client.cpp
  allocator/offset_allocator.cpp)

if(HAVE_SPDK)
  list(APPEND UMBP_CORE_SRCS
    storage/spdk/spdk_env.cpp
    storage/spdk_ssd_tier.cpp)
endif()

add_library(umbp_core STATIC ${UMBP_CORE_SRCS})
target_include_directories(umbp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(umbp_core PRIVATE pthread)

if(HAVE_SPDK)
  target_compile_definitions(umbp_core PUBLIC USE_SPDK)
  separate_arguments(SPDK_CFLAGS_LIST UNIX_COMMAND "${SPDK_CFLAGS}")
  target_compile_options(umbp_core PUBLIC ${SPDK_CFLAGS_LIST})

  separate_arguments(SPDK_LDFLAGS_LIST UNIX_COMMAND "${SPDK_LDFLAGS}")
  target_link_libraries(umbp_core PUBLIC
    -Wl,--whole-archive ${SPDK_LDFLAGS_LIST} -Wl,--no-whole-archive
    -lisal -lssl -lcrypto -lfuse3 -lnuma -ldl -lrt -luuid -lm -laio -lpthread -lelf)

  # SPDK test executables
  add_executable(test_spdk_env tests/test_spdk_env.cpp)
  target_link_libraries(test_spdk_env PRIVATE umbp_core)

  add_executable(test_spdk_ssd_tier tests/test_spdk_ssd_tier.cpp)
  target_link_libraries(test_spdk_ssd_tier PRIVATE umbp_core)

  # SPDK benchmark
  add_executable(spdk_bench benchmarks/spdk_bench.cpp)
  target_link_libraries(spdk_bench PRIVATE umbp_core)
endif()

# Python bindings are registered in src/pybind/pybind_umbp.cpp (compiled into
# libmori_pybinds when BUILD_UMBP=ON)

# C++ unit tests
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
