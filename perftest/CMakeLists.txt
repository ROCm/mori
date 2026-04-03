# Optional performance / benchmark executables for mori.
# Enable with: cmake -DBUILD_PERFTEST=ON ...
#
# Reuses include paths from examples (e.g. utils/args_parser.cpp). Add sources
# under this directory and register targets below.

# --- Shmem-style perftest: HIP sources, device kernels, links mori_shmem ---
function(add_mori_perftest_shmem name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_mori_perftest_shmem(${name}): SOURCES required")
  endif()
  add_executable(${name} ${ARG_SOURCES})
  set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE HIP)
  target_link_libraries(${name} PRIVATE mori_shmem mori_application hip::host
                                        hip::device ${ARG_LIBS})
  target_include_directories(
    ${name} PRIVATE ${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_SOURCE_DIR}/examples/utils)
  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${name} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(ARG_INCLUDES)
    target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
  endif()
endfunction()

# --- Host C++ perftest: no HIP language forced (add .hip/.cpp kernels if needed) ---
function(add_mori_perftest_app name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_mori_perftest_app(${name}): SOURCES required")
  endif()
  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE mori_application hip::host hip::device
                                        ${ARG_LIBS})
  target_include_directories(
    ${name} PRIVATE ${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_SOURCE_DIR}/examples/utils)
  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${name} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(ARG_INCLUDES)
    target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Registered perftest targets
# ---------------------------------------------------------------------------

add_mori_perftest_app(perftest SOURCES shmem/demo.cpp)

# More examples:
# add_mori_perftest_shmem(put_bw SOURCES device/put_bw.cpp)
# add_mori_perftest_app(host_bw SOURCES host/bw.cpp
#                       ${CMAKE_SOURCE_DIR}/examples/utils/args_parser.cpp)
#
# Pure C++ sources with __global__ kernels: use add_mori_perftest_shmem, or
# add_mori_perftest_app then set_source_files_properties(... LANGUAGE HIP).
# ---------------------------------------------------------------------------
