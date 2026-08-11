set(MORI_ALLOCATOR_SOURCES torch_allocator.cpp)

# Loaded by torch via dlopen (CUDAPluggableAllocator), not linked against by mori itself.
add_library(mori_torch_allocator SHARED ${MORI_ALLOCATOR_SOURCES})

target_include_directories(mori_torch_allocator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_torch_allocator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mori_torch_allocator PUBLIC mori_shmem hip::host)

set_target_properties(
  mori_torch_allocator
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
