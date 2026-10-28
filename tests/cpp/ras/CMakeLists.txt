# The RAS API with RAS built out. It never links mori_ras, so it is built either way.
add_executable(test_ras_built_out test_built_out.cpp)
target_include_directories(test_ras_built_out PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME ras_built_out COMMAND test_ras_built_out)

if(NOT BUILD_RAS)
  return()
endif()

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
mori_ras_test(identity)
mori_ras_test(config)
mori_ras_test(runtime)
mori_ras_test(fork)
mori_ras_test(links)
