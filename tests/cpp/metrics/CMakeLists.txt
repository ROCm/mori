add_executable(test_prometheus_metrics_server
               test_prometheus_metrics_server.cpp)
target_link_libraries(test_prometheus_metrics_server PRIVATE mori_metrics
                                                             gtest_main)
add_test(NAME prometheus_metrics_server COMMAND test_prometheus_metrics_server)
