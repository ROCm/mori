# KIWI queue/invoke core and the proxy-thread CPU pinning used by the KIWI EP
# proxy. Self-contained: links no other mori target.
find_package(hsa-runtime64 REQUIRED)

set(MORI_KIWI_SOURCES
    queue/device_to_host_queue.cpp
    queue/mpsc_device_to_host_queue.cpp
    queue/host_to_device_queue.cpp
    queue/mpmc_host_to_device_queue.cpp
    queue/host_memory.cpp
    topology/topology.cpp)
set_source_files_properties(${MORI_KIWI_SOURCES} PROPERTIES LANGUAGE HIP)

add_library(mori_kiwi STATIC ${MORI_KIWI_SOURCES})
target_include_directories(mori_kiwi PUBLIC ${CMAKE_SOURCE_DIR}/include
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/queue)
target_link_libraries(mori_kiwi PUBLIC hip::host hsa-runtime64::hsa-runtime64)
set_target_properties(mori_kiwi PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17
                                           HIP_STANDARD 17)
