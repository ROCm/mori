# MORI RAS tests. mori_ras_test(foo) builds test_foo.cpp as target test_ras_foo, so names don't
# collide with the IO tests (tests/cpp/io/test_protocol.cpp is MORI-IO's control plane).
function(mori_ras_test name)
  add_executable(test_ras_${name} test_${name}.cpp ${ARGN})
  target_include_directories(test_ras_${name} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(test_ras_${name} PRIVATE mori_ras gtest_main)
  add_test(NAME ras_${name} COMMAND test_ras_${name})
endfunction()

mori_ras_test(protocol)
mori_ras_test(bounded_queues)
mori_ras_test(link_timing)
