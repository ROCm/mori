# KIWI EP dispatch/combine kernels and their Python module. Self-contained:
# depends only on mori_kiwi.
set(MORI_OPS_KIWI_SOURCES
    comm/context.hip.cpp
    comm/transport_callbacks.hip.cpp
    all2all.hip.cpp
    ep_context.hip.cpp
    dispatch.hip.cpp
    combine.hip.cpp
    ep_op.hip.cpp)
set_source_files_properties(${MORI_OPS_KIWI_SOURCES} PROPERTIES LANGUAGE HIP)

add_library(mori_ops_kiwi SHARED ${MORI_OPS_KIWI_SOURCES})
target_include_directories(mori_ops_kiwi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mori_ops_kiwi PUBLIC mori_kiwi)
set_target_properties(
  mori_ops_kiwi
  PROPERTIES HIP_STANDARD 17
             CXX_STANDARD 17
             BUILD_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             INSTALL_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
install(TARGETS mori_ops_kiwi LIBRARY DESTINATION lib)

if(BUILD_PYBINDS)
  if(NOT DEFINED PYTHON_EXECUTABLE)
    execute_process(
      COMMAND python3 -c "import sys; print(sys.executable)"
      OUTPUT_VARIABLE PYTHON_EXECUTABLE
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE MORI_KIWI_PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _mori_kiwi_pybind11_result)
  if(NOT _mori_kiwi_pybind11_result EQUAL 0)
    message(FATAL_ERROR "BUILD_OPS_KIWI: failed to find pybind11 include dir")
  endif()

  add_library(mori_kiwi_ep_pybinds SHARED pybind_kiwi_ep.cpp)
  target_include_directories(mori_kiwi_ep_pybinds PRIVATE ${Python_INCLUDE_DIRS}
                                                          ${MORI_KIWI_PYBIND11_INCLUDE_DIR})
  target_link_libraries(mori_kiwi_ep_pybinds PRIVATE mori_ops_kiwi)
  set_target_properties(
    mori_kiwi_ep_pybinds
    PROPERTIES CXX_STANDARD 17
               BUILD_RPATH "$ORIGIN;${ROCM_PATH}/lib"
               INSTALL_RPATH "$ORIGIN;${ROCM_PATH}/lib"
               BUILD_WITH_INSTALL_RPATH TRUE)
endif()
