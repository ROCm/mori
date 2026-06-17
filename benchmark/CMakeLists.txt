# Optional performance / benchmark executables for mori. Enable with: cmake
# -DBUILD_PERFTEST=ON ...
#
# Reuses include paths from examples (e.g. utils/args_parser.cpp). Add sources
# under this directory and register targets below.

# --- Shmem-style benchmark: HIP sources, device kernels, links mori_shmem ---
function(add_mori_benchmark_shmem name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_mori_benchmark_shmem(${name}): SOURCES required")
  endif()
  add_executable(${name} ${ARG_SOURCES})
  set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE HIP)
  target_link_libraries(${name} PRIVATE mori_shmem mori_application hip::host
                                        hip::device ${ARG_LIBS})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include
                                             ${CMAKE_SOURCE_DIR}/examples/utils)
  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${name} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(ARG_INCLUDES)
    target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
  endif()
endfunction()

# --- Host C++ benchmark: no HIP language forced (add .hip/.cpp kernels if
# needed) ---
function(add_mori_benchmark_app name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_mori_benchmark_app(${name}): SOURCES required")
  endif()
  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE mori_application hip::host hip::device
                                        ${ARG_LIBS})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include
                                             ${CMAKE_SOURCE_DIR}/examples/utils)
  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${name} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(ARG_INCLUDES)
    target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
  endif()
endfunction()

# --- CCO benchmark: HIP kernel TU + a shared host-CXX control-plane lib ---
# CCO makes `ccoComm` host-only (opaque under __HIPCC__). util.cpp touches its
# members, so it must compile as plain CXX, never as HIP. Putting it in its own
# CXX static library guarantees that regardless of the per-target HIP language of
# the kernel TUs. Each benchmark = one HIP kernel .cpp + this lib.
function(add_mori_benchmark_cco name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_mori_benchmark_cco(${name}): SOURCES required")
  endif()
  add_executable(${name} ${ARG_SOURCES})
  set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE HIP)
  target_link_libraries(${name} PRIVATE cco_bench_util mori_cco mori_application
                                        ibverbs hip::host hip::device ${ARG_LIBS})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include
                                             ${CMAKE_CURRENT_SOURCE_DIR}/cco)
  if(DEFINED GPU_TARGETS)
    set_target_properties(${name} PROPERTIES HIP_ARCHITECTURES "${GPU_TARGETS}")
  endif()
  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${name} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(ARG_INCLUDES)
    target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Registered benchmark targets
# ---------------------------------------------------------------------------

if(WITH_MPI)
  # libmori_application uses std::filesystem; GNU libstdc++ needs explicit
  # -lstdc++fs.
  add_mori_benchmark_shmem(shmem_p2p_put_bw SOURCES shmem/p2p_put_bw.cpp
                           shmem/util.cpp LIBS stdc++fs)
  # Host-only for now, but shmem.hpp pulls in symbols from mori_shmem
  # (ShmemMpiInit, etc.).
  add_mori_benchmark_shmem(shmem_p2p_put_latency SOURCES shmem/p2p_put_latency.cpp
                           shmem/util.cpp LIBS stdc++fs)
  add_mori_benchmark_shmem(shmem_p2p_get_bw SOURCES shmem/p2p_get_bw.cpp
                           shmem/util.cpp LIBS stdc++fs)
  add_mori_benchmark_shmem(shmem_p2p_get_latency SOURCES shmem/p2p_get_latency.cpp
                           shmem/util.cpp LIBS stdc++fs)
endif()

# --- CCO p2p benchmarks (LSA + IGBDA transports, MPI 2-rank launcher) ---
if(WITH_MPI AND BUILD_CCO)
  # Host control-plane (MPI + ccoComm lifecycle) as a plain CXX static lib, so it
  # never gets the kernel TUs' HIP language (which would make ccoComm opaque).
  add_library(cco_bench_util STATIC cco/util.cpp)
  set_source_files_properties(cco/util.cpp PROPERTIES LANGUAGE CXX)
  target_include_directories(cco_bench_util PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/cco
                                                    ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(cco_bench_util PRIVATE mori_cco mori_application
                                               hip::host ${MPI_CXX_LIBRARIES})
  set_target_properties(cco_bench_util PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_mori_benchmark_cco(cco_p2p_put_bw SOURCES cco/p2p_put_bw.cpp
                         LIBS stdc++fs ${MPI_CXX_LIBRARIES}
                         INCLUDES ${MPI_CXX_INCLUDE_DIRS})
  add_mori_benchmark_cco(cco_p2p_put_latency SOURCES cco/p2p_put_latency.cpp
                         LIBS stdc++fs ${MPI_CXX_LIBRARIES}
                         INCLUDES ${MPI_CXX_INCLUDE_DIRS})
  add_mori_benchmark_cco(cco_p2p_get_bw SOURCES cco/p2p_get_bw.cpp
                         LIBS stdc++fs ${MPI_CXX_LIBRARIES}
                         INCLUDES ${MPI_CXX_INCLUDE_DIRS})
  add_mori_benchmark_cco(cco_p2p_get_latency SOURCES cco/p2p_get_latency.cpp
                         LIBS stdc++fs ${MPI_CXX_LIBRARIES}
                         INCLUDES ${MPI_CXX_INCLUDE_DIRS})
elseif(WITH_MPI AND NOT BUILD_CCO)
  message(WARNING
    "BUILD_BENCHMARK=ON with BUILD_CCO=OFF: CCO p2p benchmark targets skipped")
endif()
