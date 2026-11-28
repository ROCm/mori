set(MORI_CCO_SOURCES cco_init.cpp)

add_library(mori_cco SHARED ${MORI_CCO_SOURCES})

target_include_directories(mori_cco PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_cco PUBLIC ${CMAKE_SOURCE_DIR})

# SDMA compile switch. PUBLIC so anything linking mori_cco and including cco.hpp
# sees the same BUILD_CCO_SDMA value the library was built with (the header
# defaults it to 0). The JIT device wrapper (compiled out-of-tree by bitcode.py)
# must pass a matching -DBUILD_CCO_SDMA to stay ABI-consistent.
if(BUILD_CCO_SDMA)
  target_compile_definitions(mori_cco PUBLIC BUILD_CCO_SDMA=1)
else()
  target_compile_definitions(mori_cco PUBLIC BUILD_CCO_SDMA=0)
endif()

# Absorb the application layer statically so libmori_cco.so has NO runtime
# dependency on libmori_application.so: --whole-archive pulls in the full
# archive. --exclude-libs,ALL then marks every symbol coming from a static
# archive (mori_application_static, plus the absorbed hsakmt / spdlog / builtins)
# as local, so none are re-exported (no symbol leakage / interposition). cco's
# own cco* symbols live in this target's object file, not an archive, so they
# stay exported. The static archive's INTERFACE brings the external link deps
# (hip/dl/hsa/hsakmt[/mpi]); the ibverbs dlopen shim is embedded, so cco
# links neither libmori_application.so nor libibverbs directly.
target_link_libraries(mori_cco PRIVATE
                      -Wl,--whole-archive mori_application_static -Wl,--no-whole-archive)
target_link_options(mori_cco PRIVATE LINKER:--exclude-libs,ALL)

target_link_libraries(mori_cco PUBLIC mori_logging hip::host)

# ccoProxy host engine (gfx942/gfx950 only); MORI_CCO_PROXY marks it built in.
# Without it, cco_proxy_engine_stub.cpp provides the entry points.
set(MORI_CCO_PROXY_BUILT OFF)
if(BUILD_CCO_PROXY)
  set(MORI_CCO_PROXY_HIP_ARCHITECTURES)
  foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
    if(_arch MATCHES "^gfx(942|950)(:|$)")
      list(APPEND MORI_CCO_PROXY_HIP_ARCHITECTURES ${_arch})
    endif()
  endforeach()
  if(NOT MORI_CCO_PROXY_HIP_ARCHITECTURES)
    message(STATUS "ccoProxy: no gfx942/gfx950 in GPU_TARGETS; host engine not built")
    set(BUILD_CCO_PROXY OFF PARENT_SCOPE)
  else()
    set(MORI_CCO_PROXY_SOURCES
        cco_proxy_engine.hip.cpp proxy/device_to_host_queue.cpp
        proxy/mpsc_device_to_host_queue.cpp proxy/host_memory.cpp)
    set_source_files_properties(${MORI_CCO_PROXY_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(mori_cco_proxy OBJECT ${MORI_CCO_PROXY_SOURCES})
    target_include_directories(mori_cco_proxy PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}
                                                      ${CMAKE_CURRENT_SOURCE_DIR}/proxy)
    # HSA comes with mori_application_static, which mori_cco absorbs.
    target_link_libraries(mori_cco_proxy PRIVATE mori_logging hip::host)
    set_target_properties(
      mori_cco_proxy PROPERTIES HIP_ARCHITECTURES "${MORI_CCO_PROXY_HIP_ARCHITECTURES}"
                                POSITION_INDEPENDENT_CODE ON HIP_STANDARD 17)
    target_sources(mori_cco PRIVATE $<TARGET_OBJECTS:mori_cco_proxy>)
    target_compile_definitions(mori_cco PUBLIC MORI_CCO_PROXY=1)
    set(MORI_CCO_PROXY_BUILT ON)
  endif()
endif()
if(NOT MORI_CCO_PROXY_BUILT)
  target_sources(mori_cco PRIVATE cco_proxy_engine_stub.cpp)
endif()

set_target_properties(
  mori_cco
  PROPERTIES BUILD_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             INSTALL_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
