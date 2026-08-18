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
  # libmori_cco.so is self-contained (absorbs the application layer).
  target_link_libraries(${name} PRIVATE cco_bench_util mori_cco hip::host
                                        hip::device ${ARG_LIBS})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include
                                             ${CMAKE_CURRENT_SOURCE_DIR}/cco)
  # $ORIGIN-relative rpath (kept alongside the build-tree rpath) so the binaries
  # resolve libmori_*.so after `BUILD_BENCHMARK=ON pip install .` copies them to
  # site-packages/mori/benchmarks/cco/ (the libs live at site-packages/mori/).
  set_target_properties(${name} PROPERTIES BUILD_RPATH "$ORIGIN/../..;${ROCM_PATH}/lib")
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
  add_mori_benchmark_shmem(p2p_put_bw SOURCES shmem/p2p_put_bw.cpp
                           shmem/util.cpp LIBS stdc++fs)
  # Host-only for now, but shmem.hpp pulls in symbols from mori_shmem
  # (ShmemMpiInit, etc.).
  add_mori_benchmark_shmem(p2p_put_latency SOURCES shmem/p2p_put_latency.cpp
                           shmem/util.cpp LIBS stdc++fs)
  add_mori_benchmark_shmem(p2p_get_bw SOURCES shmem/p2p_get_bw.cpp
                           shmem/util.cpp LIBS stdc++fs)
  add_mori_benchmark_shmem(p2p_get_latency SOURCES shmem/p2p_get_latency.cpp
                           shmem/util.cpp LIBS stdc++fs)
endif()

# --- CCO p2p benchmarks (LSA + IGBDA transports, MPI 2-rank launcher) ---
if(WITH_MPI AND BUILD_CCO)
  # MPI is a direct dependency of the cco benchmarks; previously it came
  # transitively via mori_application, which the benchmarks no longer link.
  find_package(MPI REQUIRED)
  # Host control-plane (MPI + ccoComm lifecycle) as a plain CXX static lib, so it
  # never gets the kernel TUs' HIP language (which would make ccoComm opaque).
  add_library(cco_bench_util STATIC cco/util.cpp)
  set_source_files_properties(cco/util.cpp PROPERTIES LANGUAGE CXX)
  target_include_directories(cco_bench_util PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/cco
                                                    ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(cco_bench_util PRIVATE mori_cco hip::host
                                               ${MPI_CXX_LIBRARIES})
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

# --- xGMI transport probe (cco/xgmi_bw) ---
# One self-contained HIP TU: it opens its own socket and talks to the HIP runtime
# directly, so it links no mori library. It calls the gfx125x tensor-DMA builtins
# (__builtin_amdgcn_tensor_load_to_lds and friends), which do not exist on other
# targets, hence the arch filter rather than an unconditional target.
#
# The sweeps under cco/xgmi_bw/tools/ compile this same TU themselves, because
# tile, pipe depth and issuer count are compile time and a sweep changes them per
# configuration. cco/xgmi_bw/tools/build_ualoe.sh owns the flag set both paths
# use; keep the compile options below in step with it, or the numbers this target
# produces stop matching the recorded tables in cco/xgmi_bw/results/.
if(DEFINED GPU_TARGETS)
  set(_xgmi_archs "")
  foreach(_arch IN LISTS GPU_TARGETS)
    if(_arch MATCHES "^gfx125")
      list(APPEND _xgmi_archs ${_arch})
    endif()
  endforeach()
  if(_xgmi_archs)
    add_executable(ualoe_bw cco/xgmi_bw/ualoe_bw.cpp)
    set_source_files_properties(cco/xgmi_bw/ualoe_bw.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(ualoe_bw PRIVATE hip::host hip::device)
    # -O3, not the library's -O2: the recorded tables were measured at -O3 and
    # the TDM issue loop is tight enough that the two do not agree.
    target_compile_options(ualoe_bw PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
    set_target_properties(ualoe_bw PROPERTIES HIP_ARCHITECTURES "${_xgmi_archs}")
  else()
    message(STATUS
      "xgmi_bw: no gfx125x in GPU_TARGETS (${GPU_TARGETS}), ualoe_bw skipped")
  endif()
endif()
