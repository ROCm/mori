set(MORI_XSHMEM_SOURCES xshmem_init.cpp xshmem_memory.cpp)

add_library(mori_xshmem SHARED ${MORI_XSHMEM_SOURCES})

target_include_directories(mori_xshmem PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_xshmem PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mori_xshmem PUBLIC mori_application mori_logging ibverbs
                                         hip::host)

set_target_properties(
  mori_xshmem
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
