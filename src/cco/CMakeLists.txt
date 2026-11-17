set(MORI_CCO_SOURCES cco_init.cpp cco_c_api.cpp)

add_library(mori_cco SHARED ${MORI_CCO_SOURCES})

# Reported by the C ABI's getVersion.
execute_process(
  COMMAND git describe --tags --always --dirty
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE MORI_CCO_VERSION_STRING
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT MORI_CCO_VERSION_STRING)
  set(MORI_CCO_VERSION_STRING "unknown")
endif()
set_source_files_properties(
  cco_c_api.cpp PROPERTIES COMPILE_DEFINITIONS
                           "MORI_VERSION_STRING=\"${MORI_CCO_VERSION_STRING}\"")

target_include_directories(mori_cco PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_cco PUBLIC ${CMAKE_SOURCE_DIR})

# SDMA compile switch. PUBLIC so anything linking mori_cco and including cco.hpp
# sees the same BUILD_CCO_SDMA value the library was built with (the header
# defaults it to 0). The JIT device wrapper (compiled out-of-tree by bitcode.py)
# must pass a matching -DBUILD_CCO_SDMA to stay ABI-consistent.
if(BUILD_CCO_SDMA)
  target_compile_definitions(mori_cco PUBLIC BUILD_CCO_SDMA=1)
else()
  target_compile_definitions(mori_cco PUBLIC BUILD_CCO_SDMA=0)
endif()

# Absorb the application layer statically so libmori_cco.so has NO runtime
# dependency on libmori_application.so: --whole-archive pulls in the full
# archive. --exclude-libs,ALL marks every symbol defined in a static archive
# (mori_application_static, plus the absorbed hsakmt / spdlog / builtins) as
# local. Weak template and inline code instantiated in this target's own objects
# is not covered by that; the version script below hides it. The static
# archive's INTERFACE brings the external link deps (hip/dl/hsa/hsakmt[/mpi]);
# the ibverbs dlopen shim is embedded, so cco links neither
# libmori_application.so nor libibverbs directly.
target_link_libraries(mori_cco PRIVATE
                      -Wl,--whole-archive mori_application_static -Wl,--no-whole-archive)
target_link_options(mori_cco PRIVATE LINKER:--exclude-libs,ALL)

# libmori_cco.so.1, with libmori_cco.so as the link-time name. The SONAME
# changes only when the C ABI major in cco_c.h does. The version script exports
# the C ABI entry and the mori::cco C++ API, and nothing else.
set(MORI_CCO_VERSION_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/mori_cco.map)
set_target_properties(mori_cco PROPERTIES SOVERSION 1 LINK_DEPENDS
                                                     ${MORI_CCO_VERSION_SCRIPT})
target_link_options(mori_cco PRIVATE
                    LINKER:--version-script=${MORI_CCO_VERSION_SCRIPT})

target_link_libraries(mori_cco PUBLIC mori_logging hip::host)

set_target_properties(
  mori_cco
  PROPERTIES BUILD_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             INSTALL_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
