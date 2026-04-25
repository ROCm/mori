# ---------------------------------------------------------------------------
# UMBP unit tests — requires GTest and umbp_common
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)

# Prevent GoogleTest from overriding compiler/linker options when built as a
# subproject.
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ---------------------------------------------------------------------------
# test_external_kv_block_index
# ---------------------------------------------------------------------------
add_executable(test_external_kv_block_index test_external_kv_block_index.cpp)

target_link_libraries(test_external_kv_block_index PRIVATE umbp_common
                                                           GTest::gtest_main)

target_compile_features(test_external_kv_block_index PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(test_external_kv_block_index)

# ---------------------------------------------------------------------------
# test_client_registry_external_kv
# ---------------------------------------------------------------------------
add_executable(test_client_registry_external_kv
               test_client_registry_external_kv.cpp)

target_link_libraries(test_client_registry_external_kv
                      PRIVATE umbp_common GTest::gtest_main)

target_compile_features(test_client_registry_external_kv PRIVATE cxx_std_17)

gtest_discover_tests(test_client_registry_external_kv)
