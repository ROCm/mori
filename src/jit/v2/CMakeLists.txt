set(MORI_JIT_SOURCES compiler.cpp plan_api.cpp toolchain.cpp util.cpp)

add_library(mori_jit SHARED ${MORI_JIT_SOURCES})

target_include_directories(mori_jit PUBLIC ${CMAKE_SOURCE_DIR}/)
target_include_directories(mori_jit PUBLIC ${CMAKE_SOURCE_DIR}/include)

# hip::host only — the HIP runtime API, no device compilation. mori_jit must
# keep building with a plain C++ compiler; see docs/MORI_JIT_V2_DESIGN.md §3.5.1.
target_link_libraries(mori_jit PUBLIC mori_logging hip::host)

# Development builds can find the kernel sources without an env var. Wheel
# installs fall back to the packaged _jit-sources next to the .so.
target_compile_definitions(
  mori_jit PRIVATE MORI_JIT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# C++20: the Cfg field-count guard needs a requires-expression (render.hpp).
set_target_properties(
  mori_jit
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)

install(TARGETS mori_jit LIBRARY DESTINATION lib)
