# KIWI queue/invoke core and the proxy-thread CPU pinning used by the KIWI EP
# proxy. Self-contained: links no other mori target.
find_package(hsa-runtime64 REQUIRED)

# KIWI device code needs wave64 and the gfx94x/gfx950 sc0/sc1 cache-policy
# bits, so the KIWI directories build only for those targets.
set(MORI_KIWI_HIP_ARCHITECTURES)
set(_mori_kiwi_skipped_archs)
foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(_arch MATCHES "^gfx(942|950)(:|$)")
    list(APPEND MORI_KIWI_HIP_ARCHITECTURES ${_arch})
  else()
    list(APPEND _mori_kiwi_skipped_archs ${_arch})
  endif()
endforeach()
if(NOT MORI_KIWI_HIP_ARCHITECTURES)
  message(
    FATAL_ERROR
      "BUILD_OPS_KIWI=ON requires gfx942 or gfx950 in GPU_TARGETS (got: ${CMAKE_HIP_ARCHITECTURES})")
endif()
if(_mori_kiwi_skipped_archs)
  message(WARNING "BUILD_OPS_KIWI: KIWI EP is not built for ${_mori_kiwi_skipped_archs}")
endif()
message(STATUS "KIWI HIP architectures: ${MORI_KIWI_HIP_ARCHITECTURES}")
# dispatch_combine_kiwi is a sibling directory and builds for the same list.
set(MORI_KIWI_HIP_ARCHITECTURES "${MORI_KIWI_HIP_ARCHITECTURES}" PARENT_SCOPE)

set(CMAKE_HIP_ARCHITECTURES "${MORI_KIWI_HIP_ARCHITECTURES}")

set(MORI_KIWI_SOURCES
    queue/device_to_host_queue.cpp
    queue/host_memory.cpp
    topology/topology.cpp)
set_source_files_properties(${MORI_KIWI_SOURCES} PROPERTIES LANGUAGE HIP)

add_library(mori_kiwi STATIC ${MORI_KIWI_SOURCES})
target_include_directories(mori_kiwi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/queue)
target_link_libraries(mori_kiwi PUBLIC hip::host hsa-runtime64::hsa-runtime64)
set_target_properties(mori_kiwi PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17
                                           HIP_STANDARD 17)
