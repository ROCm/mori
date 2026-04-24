add_library(mori_metrics SHARED prometheus_metrics_server.cpp)

target_include_directories(mori_metrics PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mori_metrics PUBLIC mori_logging)
target_link_libraries(mori_metrics PRIVATE pthread)

set_target_properties(
  mori_metrics
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
