# mori_torch_symm -- SymmetricMemory backend registered with torch as "MORI".
#
# Self-contained: plain HIP VMM plus mori's header-only hip_compat shim, so it does not
# link mori_shmem or mori_cco. It does link libtorch, which mori otherwise does not, so
# it is opt-in via -DBUILD_TORCH_SYMM=ON. setup.py enables it when torch is importable.
if(NOT BUILD_TORCH_SYMM)
  return()
endif()

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} -c
    "import torch.utils.cpp_extension as e; print(e.include_paths()[0].rsplit('/include',1)[0])"
  OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_result)
if(NOT _torch_result EQUAL 0)
  message(FATAL_ERROR "BUILD_TORCH_SYMM=ON but torch is not importable by ${PYTHON_EXECUTABLE}")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake")
find_package(Torch REQUIRED)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE MORI_SYMM_PYBIND11_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(mori_torch_symm SHARED symm_backend.cpp)

target_include_directories(
  mori_torch_symm PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}
                          ${MORI_SYMM_PYBIND11_INCLUDE} ${Python_INCLUDE_DIRS})
target_link_libraries(mori_torch_symm PRIVATE hip::host ${TORCH_LIBRARIES}
                                              ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so)
target_compile_definitions(mori_torch_symm PRIVATE USE_ROCM=1 __HIP_PLATFORM_AMD__=1)

# Loaded as a python extension module, so no "lib" prefix.
set_target_properties(
  mori_torch_symm
  PROPERTIES PREFIX ""
             CXX_STANDARD 17
             BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
