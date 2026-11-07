set(MORI_RAS_SOURCES
    config.cpp
    event_queue.cpp
    fd_registry.cpp
    heartbeat.cpp
    identity.cpp
    link.cpp
    protocol.cpp
    runtime.cpp)

set_source_files_properties(${MORI_RAS_SOURCES} PROPERTIES LANGUAGE CXX)
# Shared, never static: the runtime is a per-process singleton, so every subsystem library has
# to reach the same copy.
add_library(mori_ras SHARED ${MORI_RAS_SOURCES})

target_include_directories(mori_ras PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_include_directories(
  mori_ras PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/msgpack-c/include)
# MORI_ENABLE_RAS switches include/mori/ras/ras.hpp from its no-op stubs to the real API in
# every target that links mori_ras.
target_compile_definitions(mori_ras PUBLIC MORI_ENABLE_RAS MSGPACK_NO_BOOST)
target_link_libraries(mori_ras PUBLIC mori_logging)
target_link_libraries(mori_ras PRIVATE pthread)
set_target_properties(
  mori_ras
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
