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

if(WITH_MPI)
  # libmori_application uses std::filesystem; GNU libstdc++ needs explicit -lstdc++fs.
  add_mori_perftest_shmem(p2p_put_bw SOURCES shmem/p2p_put_bw.cpp shmem/util.cpp LIBS stdc++fs)
  # Host-only for now, but shmem.hpp pulls in symbols from mori_shmem (ShmemMpiInit, etc.).
  add_mori_perftest_shmem(p2p_put_latency SOURCES shmem/p2p_put_latency.cpp shmem/util.cpp LIBS stdc++fs)
  add_mori_perftest_shmem(ring_put_bw SOURCES shmem/ring_put_bw.cpp shmem/util.cpp LIBS stdc++fs)
  add_mori_perftest_shmem(p2p_get_bw SOURCES shmem/p2p_get_bw.cpp shmem/util.cpp LIBS stdc++fs)
  add_mori_perftest_shmem(p2p_get_latency SOURCES shmem/p2p_get_latency.cpp shmem/util.cpp LIBS stdc++fs)
endif()
