# Single-process multi-thread test
add_executable(test_xshmem_host test_xshmem_host.cpp)
set_source_files_properties(test_xshmem_host.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(test_xshmem_host PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                     ${CMAKE_SOURCE_DIR})
target_link_libraries(test_xshmem_host PRIVATE mori_xshmem mori_application
                                                mori_logging ibverbs pthread)
set_target_properties(
  test_xshmem_host PROPERTIES SKIP_BUILD_RPATH FALSE
                              BUILD_WITH_INSTALL_RPATH FALSE
                              INSTALL_RPATH_USE_LINK_PATH TRUE)
add_test(NAME xshmem_host_api COMMAND test_xshmem_host)

# Multi-process test: auto-detects MPI or falls back to fork+socket
add_executable(test_xshmem_multiprocess test_xshmem_multiprocess.cpp)
set_source_files_properties(test_xshmem_multiprocess.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(test_xshmem_multiprocess PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                             ${CMAKE_SOURCE_DIR})
target_link_libraries(test_xshmem_multiprocess PRIVATE mori_xshmem mori_application
                                                        mori_logging ibverbs)
if(WITH_MPI)
  target_compile_definitions(test_xshmem_multiprocess PRIVATE MORI_WITH_MPI)
  target_link_libraries(test_xshmem_multiprocess PRIVATE ${MPI_CXX_LIBRARIES})
  target_include_directories(test_xshmem_multiprocess PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()
set_target_properties(
  test_xshmem_multiprocess PROPERTIES SKIP_BUILD_RPATH FALSE
                                      BUILD_WITH_INSTALL_RPATH FALSE
                                      INSTALL_RPATH_USE_LINK_PATH TRUE)

# ctest: fork mode (always available)
add_test(NAME xshmem_multiprocess COMMAND test_xshmem_multiprocess)

# ctest: MPI mode (if MPI enabled)
if(WITH_MPI)
  add_test(NAME xshmem_mpi
           COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root ${MPIEXEC_NUMPROC_FLAG} 8
                   $<TARGET_FILE:test_xshmem_multiprocess>)
endif()
