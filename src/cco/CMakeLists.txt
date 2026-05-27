set(MORI_CCO_SOURCES cco_init.cpp cco_memory.cpp)

add_library(mori_cco SHARED ${MORI_CCO_SOURCES})

target_include_directories(mori_cco PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_cco PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mori_cco PUBLIC mori_application mori_logging ibverbs
                                      hip::host)

set_target_properties(
  mori_cco
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
