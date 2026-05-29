# Single-process multi-thread test
add_executable(test_cco_host test_cco_host.cpp)
set_source_files_properties(test_cco_host.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(test_cco_host PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                 ${CMAKE_SOURCE_DIR})
target_link_libraries(test_cco_host PRIVATE mori_cco mori_application
                                            mori_logging ibverbs pthread)
set_target_properties(
  test_cco_host
  PROPERTIES SKIP_BUILD_RPATH FALSE
             BUILD_WITH_INSTALL_RPATH FALSE
             INSTALL_RPATH_USE_LINK_PATH TRUE)
add_test(NAME cco_host_api COMMAND test_cco_host)

# Multi-process test: auto-detects MPI or falls back to fork+socket
add_executable(test_cco_multiprocess test_cco_multiprocess.cpp)
set_source_files_properties(test_cco_multiprocess.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(
  test_cco_multiprocess PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_link_libraries(test_cco_multiprocess PRIVATE mori_cco mori_application
                                                    mori_logging ibverbs)
if(WITH_MPI)
  target_compile_definitions(test_cco_multiprocess PRIVATE MORI_WITH_MPI)
  target_link_libraries(test_cco_multiprocess PRIVATE ${MPI_CXX_LIBRARIES})
  target_include_directories(test_cco_multiprocess
                             PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()
set_target_properties(
  test_cco_multiprocess
  PROPERTIES SKIP_BUILD_RPATH FALSE
             BUILD_WITH_INSTALL_RPATH FALSE
             INSTALL_RPATH_USE_LINK_PATH TRUE)

# ctest: fork mode (always available)
add_test(NAME cco_multiprocess COMMAND test_cco_multiprocess)

# ctest: MPI mode (if MPI enabled)
if(WITH_MPI)
  add_test(
    NAME cco_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root ${MPIEXEC_NUMPROC_FLAG} 8
            $<TARGET_FILE:test_cco_multiprocess>)
endif()

# GDA connection mode test
add_executable(test_cco_gda_modes test_cco_gda_modes.cpp)
set_source_files_properties(test_cco_gda_modes.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(test_cco_gda_modes PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                          ${CMAKE_SOURCE_DIR})
target_link_libraries(test_cco_gda_modes PRIVATE mori_cco mori_application
                                                     mori_logging ibverbs pthread)
set_target_properties(
  test_cco_gda_modes PROPERTIES SKIP_BUILD_RPATH FALSE
                                   BUILD_WITH_INSTALL_RPATH FALSE
                                   INSTALL_RPATH_USE_LINK_PATH TRUE)
add_test(NAME cco_gda_modes COMMAND test_cco_gda_modes)

# GDA device API test (requires HIP for __global__ kernel)
add_executable(test_cco_gda_device test_cco_gda_device.cpp)
set_source_files_properties(test_cco_gda_device.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(test_cco_gda_device PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                           ${CMAKE_SOURCE_DIR})
target_link_libraries(test_cco_gda_device PRIVATE mori_cco mori_application
                                                      mori_logging ibverbs hip::host hip::device)
if(WITH_MPI)
  target_compile_definitions(test_cco_gda_device PRIVATE MORI_WITH_MPI)
  target_link_libraries(test_cco_gda_device PRIVATE ${MPI_CXX_LIBRARIES})
  target_include_directories(test_cco_gda_device PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()
if(DEFINED GPU_TARGETS)
  set_target_properties(test_cco_gda_device PROPERTIES HIP_ARCHITECTURES
                                                           "${GPU_TARGETS}")
endif()
set_target_properties(
  test_cco_gda_device PROPERTIES SKIP_BUILD_RPATH FALSE
                                    BUILD_WITH_INSTALL_RPATH FALSE
                                    INSTALL_RPATH_USE_LINK_PATH TRUE)

# ctest: fork mode
add_test(NAME cco_gda_device COMMAND test_cco_gda_device)

# ctest: MPI mode
if(WITH_MPI)
  add_test(NAME cco_gda_device_mpi
           COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root ${MPIEXEC_NUMPROC_FLAG} 8
                   $<TARGET_FILE:test_cco_gda_device>)
endif()
