add_executable(test_prometheus_metrics_server
               test_prometheus_metrics_server.cpp)
target_link_libraries(test_prometheus_metrics_server PRIVATE mori_metrics
                                                             gtest_main)
add_test(NAME prometheus_metrics_server COMMAND test_prometheus_metrics_server)

# The common layer (identities, MetricPublisher, Registry). Needs no hardware.
add_executable(test_metrics_core test_metrics_core.cpp)
target_link_libraries(test_metrics_core PRIVATE mori_metrics gtest_main)
add_test(NAME metrics_core COMMAND test_metrics_core)

# The traffic counters of each module. Cases that need hardware skip without it.
if(BUILD_IO)
  add_executable(test_traffic_metrics test_traffic_metrics.cpp)
  target_include_directories(test_traffic_metrics PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(
    test_traffic_metrics PRIVATE mori_application mori_io mori_metrics hip::host
                                 gtest_main)
  add_test(NAME traffic_metrics COMMAND test_traffic_metrics)
endif()
