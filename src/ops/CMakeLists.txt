set(MORI_OPS_SOURCES
  dispatch_combine/dispatch_combine.cpp
  dispatch_combine/internode_v1.cpp
  dispatch_combine/low_latency_async.cpp
  cast.cpp)

# These files contain __device__/__global__ code and must be compiled with hipcc
set_source_files_properties(${MORI_OPS_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(mori_ops ${MORI_OPS_SOURCES})

target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(mori_ops PRIVATE "-fgpu-rdc" "-mcode-object-version=5")
target_link_options(mori_ops PRIVATE "-fgpu-rdc" "-mcode-object-version=5")
target_link_libraries(
  mori_ops
  PUBLIC mori_application mori_shmem mori_logging hip::host)

if(ENABLE_PROFILER)
  target_link_libraries(mori_ops PUBLIC mori_profiler_interface)
endif()
