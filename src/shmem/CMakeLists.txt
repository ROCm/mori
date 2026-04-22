set(MORI_SHMEM_SOURCES init.cpp runtime.cpp memory.cpp)

# All shmem source files are now pure CXX (device code moved to JIT)
add_library(mori_shmem SHARED ${MORI_SHMEM_SOURCES})

target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_shmem PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mori_shmem PUBLIC mori_application mori_logging ibverbs
                                        hip::host)

if(MORI_MULTITHREAD_SUPPORT)
  target_compile_definitions(mori_shmem PUBLIC MORI_MULTITHREAD_SUPPORT)
  message(STATUS "mori_shmem: MORI_MULTITHREAD_SUPPORT enabled")
endif()

set_target_properties(
  mori_shmem
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
