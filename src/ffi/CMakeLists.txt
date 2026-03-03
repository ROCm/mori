add_library(mori_xla_ffi_ops SHARED mori_xla_ffi_ops.cpp)

target_include_directories(
  mori_xla_ffi_ops
  PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/xla_ffi
          ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_SOURCE_DIR}/)

target_link_libraries(
  mori_xla_ffi_ops
  PRIVATE mori_ops mori_shmem mori_application mori_logging
          hip::host hip::device)

set_target_properties(
  mori_xla_ffi_ops
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
