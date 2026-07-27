set(MORI_CCO_SOURCES cco_init.cpp)

add_library(mori_cco SHARED ${MORI_CCO_SOURCES})

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
# archive. --exclude-libs,ALL then marks every symbol coming from a static
# archive (mori_application_static, plus the absorbed hsakmt / spdlog / builtins)
# as local, so none are re-exported (no symbol leakage / interposition). cco's
# own cco* symbols live in this target's object file, not an archive, so they
# stay exported. The static archive's INTERFACE brings the external link deps
# (hip/dl/pci/hsa/hsakmt[/mpi]); the ibverbs dlopen shim is embedded, so cco
# links neither libmori_application.so nor libibverbs directly.
target_link_libraries(mori_cco PRIVATE
                      -Wl,--whole-archive mori_application_static -Wl,--no-whole-archive)
target_link_options(mori_cco PRIVATE LINKER:--exclude-libs,ALL)

target_link_libraries(mori_cco PUBLIC mori_logging hip::host)

set_target_properties(
  mori_cco
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
