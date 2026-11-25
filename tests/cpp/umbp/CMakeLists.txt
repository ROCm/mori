# CI runs every test registered below this directory (ctest from
# build/tests/cpp/umbp), whatever its name: add_test() is all a new test needs.
add_subdirectory(local)
if(TARGET umbp_common)
  add_subdirectory(distributed)
endif()
