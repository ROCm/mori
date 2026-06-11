# tests/cpp/cco — auto-discovered test binaries.
#
# every test_*.cpp in this directory is compiled and registered as a ctest.
# conventions inferred from source content (no per-file cmake plumbing):
#   - source contains __global__       → built as HIP, links hip::host/hip::device
#                                        and respects ${GPU_TARGETS}
#   - source references MORI_WITH_MPI  → when WITH_MPI is on, compiled with
#                                        -DMORI_WITH_MPI and gets a second
#                                        ctest entry running under mpirun
#
# ctest naming:
#   test_<x>.cpp  →  ctest "cco_<x>"  (+ "cco_<x>_mpi" when applicable)
#
# add a new test by dropping a file here. no further cmake changes needed.

set(CCO_TEST_COMMON_LIBS mori_cco mori_application mori_logging ibverbs pthread)
set(CCO_TEST_MPI_RANKS 8)

# helper: true if `needle` appears anywhere in `src`.
function(_cco_source_contains src needle out_var)
  file(READ "${src}" _content)
  string(FIND "${_content}" "${needle}" _pos)
  if(_pos GREATER -1)
    set(${out_var}
        TRUE
        PARENT_SCOPE)
  else()
    set(${out_var}
        FALSE
        PARENT_SCOPE)
  endif()
endfunction()

function(add_cco_test src)
  get_filename_component(target "${src}" NAME_WE) # e.g. test_gda_put
  string(REGEX REPLACE "^test_" "" short "${target}") # e.g. gda_put

  _cco_source_contains("${src}" "__global__" uses_hip)
  _cco_source_contains("${src}" "MORI_WITH_MPI" uses_mpi)

  add_executable(${target} ${src})
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include
                                               ${CMAKE_SOURCE_DIR})
  target_link_libraries(${target} PRIVATE ${CCO_TEST_COMMON_LIBS})

  if(uses_hip)
    set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
    target_link_libraries(${target} PRIVATE hip::host hip::device)
    if(DEFINED GPU_TARGETS)
      set_target_properties(${target} PROPERTIES HIP_ARCHITECTURES
                                                 "${GPU_TARGETS}")
    endif()
  else()
    set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
  endif()

  if(uses_mpi AND WITH_MPI)
    target_compile_definitions(${target} PRIVATE MORI_WITH_MPI)
    target_link_libraries(${target} PRIVATE ${MPI_CXX_LIBRARIES})
    target_include_directories(${target} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  endif()

  set_target_properties(
    ${target}
    PROPERTIES SKIP_BUILD_RPATH FALSE
               BUILD_WITH_INSTALL_RPATH FALSE
               INSTALL_RPATH_USE_LINK_PATH TRUE)

  add_test(NAME cco_${short} COMMAND ${target})
  if(uses_mpi AND WITH_MPI)
    add_test(
      NAME cco_${short}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root ${MPIEXEC_NUMPROC_FLAG}
              ${CCO_TEST_MPI_RANKS} $<TARGET_FILE:${target}>)
  endif()
endfunction()

file(GLOB CCO_TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")
foreach(_src ${CCO_TEST_SOURCES})
  add_cco_test("${_src}")
endforeach()
