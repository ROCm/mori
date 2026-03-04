include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

add_executable(test_umbp_types test_types.cpp)
target_link_libraries(test_umbp_types PRIVATE umbp_common gtest_main)
add_test(NAME umbp_types COMMAND test_umbp_types)

add_executable(test_umbp_client_registry test_client_registry.cpp)
target_link_libraries(test_umbp_client_registry PRIVATE umbp_common gtest_main)
add_test(NAME umbp_client_registry COMMAND test_umbp_client_registry)

add_executable(test_umbp_block_index test_block_index.cpp)
target_link_libraries(test_umbp_block_index PRIVATE umbp_common gtest_main)
add_test(NAME umbp_block_index COMMAND test_umbp_block_index)
