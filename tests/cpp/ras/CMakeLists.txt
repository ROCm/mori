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

# RAS through MORI-IO: EngineDesc compatibility, and two IOEngine processes linked by RAS.
if(BUILD_IO)
  mori_ras_test(engine_desc_compat)
  target_link_libraries(test_ras_engine_desc_compat PRIVATE mori_io)
  mori_ras_test(two_process)
  target_link_libraries(test_ras_two_process PRIVATE mori_io)

  # MORI RAS beside RCCL RAS in one process. Built only where RCCL is installed.
  find_library(MORI_RAS_RCCL_LIBRARY rccl HINTS ${ROCM_PATH}/lib)
  find_path(MORI_RAS_RCCL_INCLUDE rccl/rccl.h HINTS ${ROCM_PATH}/include)
  if(MORI_RAS_RCCL_LIBRARY AND MORI_RAS_RCCL_INCLUDE)
    mori_ras_test(rccl_coexistence)
    target_include_directories(test_ras_rccl_coexistence PRIVATE ${MORI_RAS_RCCL_INCLUDE})
    target_link_libraries(test_ras_rccl_coexistence PRIVATE mori_io ${MORI_RAS_RCCL_LIBRARY}
                                                            hip::host)
  else()
    message(STATUS "RCCL not found: test_ras_rccl_coexistence is not built")
  endif()
endif()

# Coverage-guided fuzzing of the protocol decoder. Needs clang's libFuzzer, so it is opt-in;
# test_ras_protocol replays its seed corpus with fixed mutations in every build.
option(MORI_RAS_FUZZ "Build fuzz_ras_protocol, a libFuzzer target for the RAS decoder" OFF)
if(MORI_RAS_FUZZ)
  add_executable(fuzz_ras_protocol fuzz_protocol.cpp ${CMAKE_SOURCE_DIR}/src/ras/protocol.cpp)
  target_include_directories(
    fuzz_ras_protocol PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include
                              ${CMAKE_SOURCE_DIR}/3rdparty/msgpack-c/include)
  target_compile_definitions(fuzz_ras_protocol PRIVATE MSGPACK_NO_BOOST)
  # Not UBSan too: with ROCm's clang, UBSan's source-location strings trip a false ASan ODR
  # report.
  target_compile_options(fuzz_ras_protocol PRIVATE -fsanitize=fuzzer,address)
  target_link_options(fuzz_ras_protocol PRIVATE -fsanitize=fuzzer,address)
endif()
