# v2 HIP backend, host side. The kernel body (ep_intranode_kernel.hpp) is reached
# solely through the JIT-generated TU, never by anything compiled here -- that is
# what keeps this target buildable without hipcc.
add_library(mori_ops_v2 SHARED ep_spec.cpp)

target_include_directories(mori_ops_v2 PUBLIC ${CMAKE_SOURCE_DIR}/)
target_include_directories(mori_ops_v2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mori_ops_v2 PUBLIC mori_jit mori_logging hip::host)

# C++20: the Cfg field-count guard needs a requires-expression (render.hpp).
set_target_properties(
  mori_ops_v2
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)

install(TARGETS mori_ops_v2 LIBRARY DESTINATION lib)
