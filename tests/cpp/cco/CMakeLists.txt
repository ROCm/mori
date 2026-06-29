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

# ---------------------------------------------------------------------------
# Detect IONIC CCQE support at configure time (mirrors python/mori/jit/core.py)
# Conditions: NIC is ionic, libionic has ionic_dv_create_cq_ex, and all ionic
# devices have firmware >= 1.117.5-a-58.
# ---------------------------------------------------------------------------
set(CCO_TEST_IONIC_CCQE FALSE)
if(MORI_DEVICE_NIC STREQUAL "ionic")
  find_library(_cco_ionic_lib NAMES ionic
               HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
  set(_lib_ok FALSE)
  if(_cco_ionic_lib)
    execute_process(COMMAND nm -D "${_cco_ionic_lib}"
                    OUTPUT_VARIABLE _ionic_syms ERROR_QUIET RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
      string(FIND "${_ionic_syms}" "ionic_dv_create_cq_ex" _pos)
      if(NOT _pos EQUAL -1)
        set(_lib_ok TRUE)
      endif()
    endif()
  endif()

  if(_lib_ok)
    set(_fw_ok TRUE)
    set(_found_any FALSE)
    file(GLOB _ib_devs "/sys/class/infiniband/*")
    foreach(_dev ${_ib_devs})
      execute_process(COMMAND readlink -f "${_dev}/device/driver"
                      OUTPUT_VARIABLE _drv OUTPUT_STRIP_TRAILING_WHITESPACE
                      ERROR_QUIET RESULT_VARIABLE _rc)
      if(NOT _rc EQUAL 0)
        continue()
      endif()
      get_filename_component(_drv_name "${_drv}" NAME)
      if(NOT (_drv_name MATCHES "^ionic"))
        continue()
      endif()
      set(_found_any TRUE)
      set(_fw_path "${_dev}/fw_ver")
      if(EXISTS "${_fw_path}")
        file(READ "${_fw_path}" _fw_str)
        string(STRIP "${_fw_str}" _fw_str)
        if(_fw_str MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-a-?([0-9]+)$")
          set(_maj ${CMAKE_MATCH_1})
          set(_min ${CMAKE_MATCH_2})
          set(_pat ${CMAKE_MATCH_3})
          set(_bld ${CMAKE_MATCH_4})
          # Compare against minimum 1.117.5-a-58
          if(_maj LESS 1 OR
             (_maj EQUAL 1 AND _min LESS 117) OR
             (_maj EQUAL 1 AND _min EQUAL 117 AND _pat LESS 5) OR
             (_maj EQUAL 1 AND _min EQUAL 117 AND _pat EQUAL 5 AND _bld LESS 58))
            set(_fw_ok FALSE)
          endif()
        else()
          set(_fw_ok FALSE)
        endif()
      else()
        set(_fw_ok FALSE)
      endif()
    endforeach()

    if(_found_any AND _fw_ok)
      set(CCO_TEST_IONIC_CCQE TRUE)
    endif()
  endif()

  message(STATUS "CCO test IONIC_CCQE: ${CCO_TEST_IONIC_CCQE}")
endif()

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
    # Compile-time GDA provider dispatch (CCO_GDA_DISPATCH in cco_scale_out.hpp)
    # keys off the auto-detected NIC; without this a BNXT/ionic host defaults to mlx5.
    if(MORI_DEVICE_NIC_DEFINE)
      target_compile_definitions(${target} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
    endif()
    # Override the global IONIC_CCQE: remove it, then re-add only if detected.
    # Collect all target-level defs set so far (e.g. MORI_DEVICE_NIC_*).
    get_target_property(_tgt_defs ${target} COMPILE_DEFINITIONS)
    if(NOT _tgt_defs)
      set(_tgt_defs "")
    endif()
    get_directory_property(_dir_defs COMPILE_DEFINITIONS)
    set_property(TARGET ${target} PROPERTY COMPILE_DEFINITIONS)
    foreach(_def ${_dir_defs})
      if(NOT _def STREQUAL "IONIC_CCQE")
        target_compile_definitions(${target} PRIVATE ${_def})
      endif()
    endforeach()
    foreach(_def ${_tgt_defs})
      if(NOT _def STREQUAL "IONIC_CCQE")
        target_compile_definitions(${target} PRIVATE ${_def})
      endif()
    endforeach()
    if(CCO_TEST_IONIC_CCQE)
      target_compile_definitions(${target} PRIVATE IONIC_CCQE)
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
