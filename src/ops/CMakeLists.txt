set(MORI_OPS_SOURCES
  dispatch_combine/dispatch_combine.cpp)

# All ops source files are now pure CXX (kernel launch moved to Python JIT)
add_library(mori_ops SHARED ${MORI_OPS_SOURCES})

target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(
  mori_ops
  PUBLIC mori_application mori_shmem mori_logging hip::host)

set_target_properties(
  mori_ops
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)

if(ENABLE_PROFILER)
  target_link_libraries(mori_ops PUBLIC mori_profiler_interface)
endif()
