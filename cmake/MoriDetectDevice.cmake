# MoriDetectDevice.cmake — Auto-detect GPU architecture and RDMA NIC type.
#
# Provides:
#   mori_detect_device_config()
#     Sets the following variables in the caller's scope:
#       MORI_GPU_ARCH          — GPU architecture string (e.g. "gfx942")
#       MORI_DEVICE_NIC        — NIC type: "mlx5", "bnxt", or "ionic"
#       MORI_DEVICE_NIC_DEFINE — Compile definition (e.g. "MORI_DEVICE_NIC_BNXT"),
#                                empty for mlx5 (the default provider)
#
#   mori_add_device_target(<target>)
#     Convenience function that applies MORI_DEVICE_NIC_DEFINE and include dirs
#     to an existing HIP target. Call mori_detect_device_config() first.
#
# Usage in an external project:
#   include(/path/to/MoriDetectDevice.cmake)
#   mori_detect_device_config()
#   add_executable(my_app my_kernel.hip)
#   set_source_files_properties(my_kernel.hip PROPERTIES LANGUAGE HIP)
#   target_link_libraries(my_app mori::shmem hip::device)
#   mori_add_device_target(my_app)

include_guard(GLOBAL)

set(_MORI_SUPPORTED_ARCHS "gfx942;gfx950")

# ---------------------------------------------------------------------------
# GPU architecture detection
# ---------------------------------------------------------------------------
function(_mori_detect_gpu_arch out_var)
  # 1. Env override
  if(DEFINED ENV{MORI_GPU_ARCHS})
    foreach(_arch ${_MORI_SUPPORTED_ARCHS})
      string(FIND "$ENV{MORI_GPU_ARCHS}" "${_arch}" _pos)
      if(NOT _pos EQUAL -1)
        message(STATUS "Mori GPU arch: ${_arch} (from MORI_GPU_ARCHS env)")
        set(${out_var} "${_arch}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()

  # 2. GPU_TARGETS / AMDGPU_TARGETS already set (e.g. by ROCm CMake or user)
  if(DEFINED GPU_TARGETS AND NOT GPU_TARGETS STREQUAL "")
    list(GET GPU_TARGETS 0 _arch)
    message(STATUS "Mori GPU arch: ${_arch} (from GPU_TARGETS)")
    set(${out_var} "${_arch}" PARENT_SCOPE)
    return()
  endif()
  if(DEFINED AMDGPU_TARGETS AND NOT AMDGPU_TARGETS STREQUAL "")
    list(GET AMDGPU_TARGETS 0 _arch)
    message(STATUS "Mori GPU arch: ${_arch} (from AMDGPU_TARGETS)")
    set(${out_var} "${_arch}" PARENT_SCOPE)
    return()
  endif()

  # 3. rocm_agent_enumerator
  set(_rocm_path "$ENV{ROCM_PATH}")
  if(NOT _rocm_path)
    set(_rocm_path "/opt/rocm")
  endif()
  set(_enumerator "${_rocm_path}/bin/rocm_agent_enumerator")
  if(EXISTS "${_enumerator}")
    execute_process(
      COMMAND "${_enumerator}"
      OUTPUT_VARIABLE _agents OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
      string(REPLACE "\n" ";" _agent_list "${_agents}")
      foreach(_line ${_agent_list})
        string(STRIP "${_line}" _line)
        foreach(_arch ${_MORI_SUPPORTED_ARCHS})
          if(_line STREQUAL _arch)
            message(STATUS "Mori GPU arch: ${_arch} (from rocm_agent_enumerator)")
            set(${out_var} "${_arch}" PARENT_SCOPE)
            return()
          endif()
        endforeach()
      endforeach()
    endif()
  endif()

  # 4. rocminfo
  find_program(_rocminfo rocminfo)
  if(_rocminfo)
    execute_process(
      COMMAND "${_rocminfo}"
      OUTPUT_VARIABLE _rocminfo_out OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
      foreach(_arch ${_MORI_SUPPORTED_ARCHS})
        string(FIND "${_rocminfo_out}" "${_arch}" _pos)
        if(NOT _pos EQUAL -1)
          message(STATUS "Mori GPU arch: ${_arch} (from rocminfo)")
          set(${out_var} "${_arch}" PARENT_SCOPE)
          return()
        endif()
      endforeach()
    endif()
  endif()

  # 5. AMDGPU_TARGETS env
  if(DEFINED ENV{AMDGPU_TARGETS})
    foreach(_arch ${_MORI_SUPPORTED_ARCHS})
      string(FIND "$ENV{AMDGPU_TARGETS}" "${_arch}" _pos)
      if(NOT _pos EQUAL -1)
        message(STATUS "Mori GPU arch: ${_arch} (from AMDGPU_TARGETS env)")
        set(${out_var} "${_arch}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()

  message(WARNING "Mori: cannot detect GPU architecture. "
    "Set GPU_TARGETS, MORI_GPU_ARCHS, or AMDGPU_TARGETS.")
  set(${out_var} "" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Device NIC detection
# Same logic as Python JIT detect_nic_type(): env > sysfs > lspci > host libs.
# ---------------------------------------------------------------------------
function(_mori_detect_device_nic out_var)
  # Find NIC libraries (needed for validation)
  find_library(_mori_bnxt_re_lib NAMES bnxt_re)
  find_library(_mori_ionic_lib NAMES ionic HINTS /lib/x86_64-linux-gnu)
  find_library(_mori_mlx5_lib NAMES mlx5)

  find_path(_mori_bnxt_dv_hdr NAMES infiniband/bnxt_re_dv.h
    HINTS /usr/include /usr/local/include)
  find_path(_mori_bnxt_hsi_hdr NAMES infiniband/bnxt_re_hsi.h
    HINTS /usr/include /usr/local/include)
  if(_mori_bnxt_dv_hdr AND _mori_bnxt_hsi_hdr)
    set(_mori_bnxt_headers TRUE)
  else()
    set(_mori_bnxt_headers FALSE)
  endif()

  macro(_mori_has_nic_lib _nic _result)
    if(${_nic} STREQUAL "bnxt" AND _mori_bnxt_re_lib AND _mori_bnxt_headers)
      set(${_result} TRUE)
    elseif(${_nic} STREQUAL "ionic" AND _mori_ionic_lib)
      set(${_result} TRUE)
    elseif(${_nic} STREQUAL "mlx5" AND _mori_mlx5_lib)
      set(${_result} TRUE)
    else()
      set(${_result} FALSE)
    endif()
  endmacro()

  # 1. Env override
  if(DEFINED ENV{MORI_DEVICE_NIC})
    string(TOLOWER "$ENV{MORI_DEVICE_NIC}" _nic)
    message(STATUS "Mori device NIC: ${_nic} (from MORI_DEVICE_NIC env)")
    set(${out_var} "${_nic}" PARENT_SCOPE)
    return()
  endif()

  # 2. /sys/class/infiniband/
  file(GLOB _ib_devices "/sys/class/infiniband/*")
  set(_bnxt 0)
  set(_ionic 0)
  set(_mlx5 0)
  foreach(_dev ${_ib_devices})
    get_filename_component(_name ${_dev} NAME)
    if(_name MATCHES "^bnxt_re")
      math(EXPR _bnxt "${_bnxt} + 1")
    elseif(_name MATCHES "^ionic")
      math(EXPR _ionic "${_ionic} + 1")
    elseif(_name MATCHES "^mlx5")
      math(EXPR _mlx5 "${_mlx5} + 1")
    else()
      execute_process(
        COMMAND readlink -f "${_dev}/device/driver"
        OUTPUT_VARIABLE _drv OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE _rc)
      if(_rc EQUAL 0)
        get_filename_component(_drv_name "${_drv}" NAME)
        if(_drv_name MATCHES "^bnxt")
          math(EXPR _bnxt "${_bnxt} + 1")
        elseif(_drv_name MATCHES "^ionic")
          math(EXPR _ionic "${_ionic} + 1")
        elseif(_drv_name MATCHES "^mlx5")
          math(EXPR _mlx5 "${_mlx5} + 1")
        endif()
      endif()
    endif()
  endforeach()

  set(_sysfs_candidates "")
  if(_mlx5 GREATER 0)
    list(APPEND _sysfs_candidates "${_mlx5}:mlx5")
  endif()
  if(_bnxt GREATER 0)
    list(APPEND _sysfs_candidates "${_bnxt}:bnxt")
  endif()
  if(_ionic GREATER 0)
    list(APPEND _sysfs_candidates "${_ionic}:ionic")
  endif()
  if(_sysfs_candidates)
    list(SORT _sysfs_candidates COMPARE NATURAL ORDER DESCENDING)
    foreach(_entry ${_sysfs_candidates})
      string(REGEX REPLACE "^[0-9]+:" "" _nic "${_entry}")
      _mori_has_nic_lib(${_nic} _has_lib)
      if(_has_lib)
        message(STATUS "Mori device NIC: ${_nic} (sysfs, mlx5=${_mlx5} bnxt=${_bnxt} ionic=${_ionic})")
        set(${out_var} "${_nic}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()

  # 3. lspci PCI vendor ID
  execute_process(
    COMMAND lspci -nn -d ::0200
    OUTPUT_VARIABLE _lspci ERROR_QUIET RESULT_VARIABLE _rc)
  if(_rc EQUAL 0 AND _lspci)
    string(REGEX MATCHALL "14e4" _b "${_lspci}")
    string(REGEX MATCHALL "1dd8" _i "${_lspci}")
    string(REGEX MATCHALL "15b3" _m "${_lspci}")
    list(LENGTH _b _bp)
    list(LENGTH _i _ip)
    list(LENGTH _m _mp)

    set(_lspci_candidates "")
    if(_mp GREATER 0)
      list(APPEND _lspci_candidates "${_mp}:mlx5")
    endif()
    if(_bp GREATER 0)
      list(APPEND _lspci_candidates "${_bp}:bnxt")
    endif()
    if(_ip GREATER 0)
      list(APPEND _lspci_candidates "${_ip}:ionic")
    endif()
    if(_lspci_candidates)
      list(SORT _lspci_candidates COMPARE NATURAL ORDER DESCENDING)
      foreach(_entry ${_lspci_candidates})
        string(REGEX REPLACE "^[0-9]+:" "" _nic "${_entry}")
        _mori_has_nic_lib(${_nic} _has_lib)
        if(_has_lib)
          message(STATUS "Mori device NIC: ${_nic} (lspci)")
          set(${out_var} "${_nic}" PARENT_SCOPE)
          return()
        endif()
      endforeach()
    endif()
  endif()

  # 4. Fallback: first available library (mlx5 preferred)
  if(_mori_mlx5_lib)
    set(_nic "mlx5")
  elseif(_mori_bnxt_re_lib AND _mori_bnxt_headers)
    set(_nic "bnxt")
  elseif(_mori_ionic_lib)
    set(_nic "ionic")
  else()
    set(_nic "mlx5")
  endif()
  message(STATUS "Mori device NIC: ${_nic} (library fallback)")
  set(${out_var} "${_nic}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Public API: mori_detect_device_config()
# ---------------------------------------------------------------------------
function(mori_detect_device_config)
  _mori_detect_gpu_arch(_gpu_arch)
  _mori_detect_device_nic(_device_nic)

  if(_device_nic STREQUAL "bnxt")
    set(_nic_define "MORI_DEVICE_NIC_BNXT")
  elseif(_device_nic STREQUAL "ionic")
    set(_nic_define "MORI_DEVICE_NIC_IONIC")
  else()
    set(_nic_define "")
  endif()

  set(MORI_GPU_ARCH "${_gpu_arch}" PARENT_SCOPE)
  set(MORI_DEVICE_NIC "${_device_nic}" PARENT_SCOPE)
  set(MORI_DEVICE_NIC_DEFINE "${_nic_define}" PARENT_SCOPE)

  message(STATUS "Mori device config: arch=${_gpu_arch}, nic=${_device_nic}, define=${_nic_define}")
endfunction()

# ---------------------------------------------------------------------------
# Public API: mori_add_device_target(<target>)
# ---------------------------------------------------------------------------
function(mori_add_device_target target)
  if(NOT DEFINED MORI_GPU_ARCH)
    message(FATAL_ERROR "Call mori_detect_device_config() before mori_add_device_target()")
  endif()

  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${target} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  target_compile_definitions(${target} PRIVATE HIP_ENABLE_WARP_SYNC_BUILTINS)

  if(MORI_GPU_ARCH)
    set_target_properties(${target} PROPERTIES HIP_ARCHITECTURES "${MORI_GPU_ARCH}")
  endif()
endfunction()
