option(UMBP_FETCH_GOOGLETEST "Fetch googletest for UMBP distributed tests" ON)

add_subdirectory(local)
if(TARGET umbp_common AND UMBP_FETCH_GOOGLETEST)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(distributed)
endif()
