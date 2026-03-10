# Find OpenSSL for SHA-256 (required by UMBP)
find_package(OpenSSL REQUIRED)

# Core static library
add_library(
  umbp_core STATIC
  block_index/block_index.cpp storage/tier_backend.cpp storage/dram_tier.cpp
  storage/ssd_tier.cpp storage/local_storage_manager.cpp umbp_client.cpp)
target_include_directories(umbp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(umbp_core PRIVATE pthread OpenSSL::Crypto)

# Python bindings are registered in src/pybind/pybind_umbp.cpp (compiled into
# libmori_pybinds when BUILD_UMBP=ON)

# C++ unit tests
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
