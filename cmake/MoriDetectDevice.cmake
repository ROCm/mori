# MoriDetectDevice.cmake — Auto-detect GPU architecture and RDMA NIC type.
#
# Provides: mori_detect_device_config() Sets the following variables in the
# caller's scope: MORI_GPU_ARCH          — GPU architecture string (e.g.
# "gfx942") MORI_DEVICE_NIC        — NIC type: "mlx5", "bnxt", or "ionic"
# MORI_DEVICE_NIC_DEFINE — Compile definition (e.g. "MORI_DEVICE_NIC_BNXT"),
# empty for mlx5 (the default provider) MORI_IONIC_CCQE_DEFINE — "IONIC_CCQE"
# when the ionic collapsed-CQE device path must be compiled in, empty otherwise
#
# mori_add_device_target(<target>) Convenience function that applies
# MORI_DEVICE_NIC_DEFINE and include dirs to an existing HIP target. Call
# mori_detect_device_config() first.
#
# Usage in an external project: include(/path/to/MoriDetectDevice.cmake)
# mori_detect_device_config() add_executable(my_app my_kernel.hip)
# set_source_files_properties(my_kernel.hip PROPERTIES LANGUAGE HIP)
# target_link_libraries(my_app mori::shmem hip::device)
# mori_add_device_target(my_app)

include_guard(GLOBAL)

set(_MORI_SUPPORTED_ARCHS "gfx942;gfx950")

set(MORI_IONIC_CCQE
    "AUTO"
    CACHE STRING
          "Ionic collapsed CQE device path: AUTO (probe firmware), ON, or OFF")
set_property(CACHE MORI_IONIC_CCQE PROPERTY STRINGS AUTO ON OFF)

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
        set(${out_var}
            "${_arch}"
            PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()

  # 1. GPU_TARGETS / AMDGPU_TARGETS already set (e.g. by ROCm CMake or user)
  if(DEFINED GPU_TARGETS AND NOT GPU_TARGETS STREQUAL "")
    list(GET GPU_TARGETS 0 _arch)
    message(STATUS "Mori GPU arch: ${_arch} (from GPU_TARGETS)")
    set(${out_var}
        "${_arch}"
        PARENT_SCOPE)
    return()
  endif()
  if(DEFINED AMDGPU_TARGETS AND NOT AMDGPU_TARGETS STREQUAL "")
    list(GET AMDGPU_TARGETS 0 _arch)
    message(STATUS "Mori GPU arch: ${_arch} (from AMDGPU_TARGETS)")
    set(${out_var}
        "${_arch}"
        PARENT_SCOPE)
    return()
  endif()

  # 1. rocm_agent_enumerator
  set(_rocm_path "$ENV{ROCM_PATH}")
  if(NOT _rocm_path)
    set(_rocm_path "/opt/rocm")
  endif()
  set(_enumerator "${_rocm_path}/bin/rocm_agent_enumerator")
  if(EXISTS "${_enumerator}")
    execute_process(
      COMMAND "${_enumerator}"
      OUTPUT_VARIABLE _agents
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
      RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
      string(REPLACE "\n" ";" _agent_list "${_agents}")
      foreach(_line ${_agent_list})
        string(STRIP "${_line}" _line)
        foreach(_arch ${_MORI_SUPPORTED_ARCHS})
          if(_line STREQUAL _arch)
            message(
              STATUS "Mori GPU arch: ${_arch} (from rocm_agent_enumerator)")
            set(${out_var}
                "${_arch}"
                PARENT_SCOPE)
            return()
          endif()
        endforeach()
      endforeach()
    endif()
  endif()

  # 1. rocminfo
  find_program(_rocminfo rocminfo)
  if(_rocminfo)
    execute_process(
      COMMAND "${_rocminfo}"
      OUTPUT_VARIABLE _rocminfo_out
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
      RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
      foreach(_arch ${_MORI_SUPPORTED_ARCHS})
        string(FIND "${_rocminfo_out}" "${_arch}" _pos)
        if(NOT _pos EQUAL -1)
          message(STATUS "Mori GPU arch: ${_arch} (from rocminfo)")
          set(${out_var}
              "${_arch}"
              PARENT_SCOPE)
          return()
        endif()
      endforeach()
    endif()
  endif()

  # 1. AMDGPU_TARGETS env
  if(DEFINED ENV{AMDGPU_TARGETS})
    foreach(_arch ${_MORI_SUPPORTED_ARCHS})
      string(FIND "$ENV{AMDGPU_TARGETS}" "${_arch}" _pos)
      if(NOT _pos EQUAL -1)
        message(STATUS "Mori GPU arch: ${_arch} (from AMDGPU_TARGETS env)")
        set(${out_var}
            "${_arch}"
            PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()

  message(WARNING "Mori: cannot detect GPU architecture. "
                  "Set GPU_TARGETS, MORI_GPU_ARCHS, or AMDGPU_TARGETS.")
  set(${out_var}
      ""
      PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Device NIC detection Same logic as Python JIT detect_nic_type(): env > sysfs >
# lspci > host libs.
# ---------------------------------------------------------------------------
function(_mori_detect_device_nic out_var)
  # Find NIC libraries (needed for validation)
  find_library(
    _mori_bnxt_re_lib
    NAMES bnxt_re bnxt_re-rdmav59 bnxt_re-rdmav34
    HINTS /usr/local/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu)
  find_library(
    _mori_ionic_lib
    NAMES ionic
    HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
  find_library(
    _mori_mlx5_lib
    NAMES mlx5
    HINTS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu)

  # bnxt headers (bnxt_re_dv.h, bnxt_re_hsi.h) are bundled in the mori source
  # tree at include/mori/core/transport/rdma/providers/bnxt/, so no system
  # header check is needed — only the userspace verbs provider library matters.

  macro(_mori_has_nic_lib _nic _result)
    if(${_nic} STREQUAL "bnxt" AND _mori_bnxt_re_lib)
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
    set(${out_var}
        "${_nic}"
        PARENT_SCOPE)
    return()
  endif()

  # 1. /sys/class/infiniband/
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
        OUTPUT_VARIABLE _drv
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        RESULT_VARIABLE _rc)
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
    list(
      SORT _sysfs_candidates
      COMPARE NATURAL
      ORDER DESCENDING)
    foreach(_entry ${_sysfs_candidates})
      string(REGEX REPLACE "^[0-9]+:" "" _nic "${_entry}")
      _mori_has_nic_lib(${_nic} _has_lib)
      if(_has_lib)
        message(
          STATUS
            "Mori device NIC: ${_nic} (sysfs, mlx5=${_mlx5} bnxt=${_bnxt} ionic=${_ionic})"
        )
        set(${out_var}
            "${_nic}"
            PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()

  # 1. lspci PCI vendor ID
  execute_process(
    COMMAND lspci -nn -d ::0200
    OUTPUT_VARIABLE _lspci
    ERROR_QUIET
    RESULT_VARIABLE _rc)
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
      list(
        SORT _lspci_candidates
        COMPARE NATURAL
        ORDER DESCENDING)
      foreach(_entry ${_lspci_candidates})
        string(REGEX REPLACE "^[0-9]+:" "" _nic "${_entry}")
        _mori_has_nic_lib(${_nic} _has_lib)
        if(_has_lib)
          message(STATUS "Mori device NIC: ${_nic} (lspci)")
          set(${out_var}
              "${_nic}"
              PARENT_SCOPE)
          return()
        endif()
      endforeach()
    endif()
  endif()

  # 1. Fallback: first available library (mlx5 preferred)
  if(_mori_mlx5_lib)
    set(_nic "mlx5")
  elseif(_mori_bnxt_re_lib)
    set(_nic "bnxt")
  elseif(_mori_ionic_lib)
    set(_nic "ionic")
  else()
    set(_nic "mlx5")
  endif()
  message(STATUS "Mori device NIC: ${_nic} (library fallback)")
  set(${out_var}
      "${_nic}"
      PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Ionic collapsed-CQE (CCQE) detection
#
# The host picks the CQ layout through IonicCcqeEnabled() in
# mori/utils/ionic_ccqe.hpp, also used by both JIT compilers. The AOT poller
# picks it at compile time (#ifdef IONIC_CCQE), so AUTO must follow the same
# policy: the provider exports ionic_dv_create_cq_ex, every IB device has a
# readable driver link, and every Ionic device runs the same firmware, at least
# 1.117.5-a-58. Disagreeing on the CQ layout can hang completion polling.
# ---------------------------------------------------------------------------
function(_mori_ionic_fw_supports_ccqe fw_ver out_var)
  set(_min_version "1;117;5;58")
  set(${out_var}
      FALSE
      PARENT_SCOPE)
  if(NOT fw_ver MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-a-?([0-9]+)$")
    return()
  endif()
  set(_ver
      "${CMAKE_MATCH_1};${CMAKE_MATCH_2};${CMAKE_MATCH_3};${CMAKE_MATCH_4}")
  foreach(_i RANGE 3)
    list(GET _ver ${_i} _have)
    list(GET _min_version ${_i} _want)
    if(_have GREATER _want)
      set(${out_var}
          TRUE
          PARENT_SCOPE)
      return()
    elseif(_have LESS _want)
      return()
    endif()
  endforeach()
  set(${out_var}
      TRUE
      PARENT_SCOPE)
endfunction()

function(_mori_detect_ionic_ccqe out_var)
  set(${out_var}
      FALSE
      PARENT_SCOPE)
  # Optional sysfs root for CPU tests of the CMake/runtime policy agreement.
  set(_ib_root "/sys/class/infiniband")
  if(ARGC GREATER 1)
    set(_ib_root "${ARGV1}")
  endif()

  if(DEFINED ENV{MORI_DISABLE_IONIC_CCQE})
    string(TOLOWER "$ENV{MORI_DISABLE_IONIC_CCQE}" _disable)
    if(_disable MATCHES "^(1|true|on|yes)$")
      message(STATUS "Mori ionic CCQE: off (MORI_DISABLE_IONIC_CCQE env)")
      return()
    endif()
  endif()

  # Userspace provider support: the collapsed CQ is created through
  # ionic_dv_create_cq_ex, which older libionic builds do not export.
  if(NOT _mori_ionic_lib)
    message(STATUS "Mori ionic CCQE: off (libionic not found)")
    return()
  endif()
  find_program(_mori_nm NAMES nm)
  if(NOT _mori_nm)
    message(
      STATUS "Mori ionic CCQE: off (nm not available to inspect libionic)")
    return()
  endif()
  execute_process(
    COMMAND "${_mori_nm}" --dynamic --defined-only "${_mori_ionic_lib}"
    OUTPUT_VARIABLE _ionic_syms
    ERROR_QUIET
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0 OR NOT _ionic_syms MATCHES "ionic_dv_create_cq_ex")
    message(
      STATUS "Mori ionic CCQE: off (libionic has no ionic_dv_create_cq_ex)")
    return()
  endif()

  # Mirror DetectIonicCcqe(): an unclassified device may be an Ionic rail.
  # Read the symlink itself, not its target; device names are not driver names.
  file(GLOB _ib_devices "${_ib_root}/*")
  set(_fw_versions "")
  foreach(_dev ${_ib_devices})
    execute_process(
      COMMAND readlink "${_dev}/device/driver"
      OUTPUT_VARIABLE _drv
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
      RESULT_VARIABLE _drv_rc)
    if(NOT _drv_rc EQUAL 0)
      message(STATUS "Mori ionic CCQE: off (unreadable driver link: ${_dev})")
      return()
    endif()
    get_filename_component(_drv_name "${_drv}" NAME)
    if(NOT _drv_name STREQUAL "ionic" AND NOT _drv_name STREQUAL "ionic_rdma")
      continue()
    endif()

    # Missing, unreadable or empty firmware must not silently drop a rail.
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E cat "${_dev}/fw_ver"
      OUTPUT_VARIABLE _fw
      ERROR_QUIET
      RESULT_VARIABLE _fw_rc)
    if(NOT _fw_rc EQUAL 0)
      message(STATUS "Mori ionic CCQE: off (unreadable firmware: ${_dev})")
      return()
    endif()
    # The shared runtime detector reads one line with std::getline.
    string(FIND "${_fw}" "\n" _newline)
    if(NOT _newline EQUAL -1)
      string(SUBSTRING "${_fw}" 0 ${_newline} _fw)
    endif()
    string(STRIP "${_fw}" _fw)
    _mori_ionic_fw_supports_ccqe("${_fw}" _fw_ok)
    if(NOT _fw_ok)
      message(STATUS "Mori ionic CCQE: off (unsupported firmware '${_fw}': ${_dev})")
      return()
    endif()
    list(APPEND _fw_versions "${_fw}")
  endforeach()

  if(NOT _fw_versions)
    message(STATUS "Mori ionic CCQE: off (no ionic fw_ver in sysfs)")
    return()
  endif()
  list(REMOVE_DUPLICATES _fw_versions)
  list(LENGTH _fw_versions _num_versions)
  if(NOT _num_versions EQUAL 1)
    message(STATUS "Mori ionic CCQE: off (mixed firmware: ${_fw_versions})")
    return()
  endif()
  list(GET _fw_versions 0 _fw)

  message(STATUS "Mori ionic CCQE: on (firmware ${_fw})")
  set(${out_var}
      TRUE
      PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Public API: mori_detect_device_config()
# ---------------------------------------------------------------------------
function(mori_detect_device_config)
  _mori_detect_gpu_arch(_gpu_arch)
  _mori_detect_device_nic(_device_nic)

  set(_ccqe_define "")
  if(_device_nic STREQUAL "bnxt")
    set(_nic_define "MORI_DEVICE_NIC_BNXT")
  elseif(_device_nic STREQUAL "ionic")
    set(_nic_define "MORI_DEVICE_NIC_IONIC")
    string(TOUPPER "${MORI_IONIC_CCQE}" _ccqe_mode)
    if(_ccqe_mode STREQUAL "AUTO")
      _mori_detect_ionic_ccqe(_ccqe_detected)
      if(_ccqe_detected)
        set(_ccqe_define "IONIC_CCQE")
      endif()
    elseif(_ccqe_mode)
      set(_ccqe_define "IONIC_CCQE")
      message(STATUS "Mori ionic CCQE: on (MORI_IONIC_CCQE=${MORI_IONIC_CCQE})")
    else()
      message(
        STATUS "Mori ionic CCQE: off (MORI_IONIC_CCQE=${MORI_IONIC_CCQE})")
    endif()
  else()
    set(_nic_define "")
  endif()

  set(MORI_GPU_ARCH
      "${_gpu_arch}"
      PARENT_SCOPE)
  set(MORI_DEVICE_NIC
      "${_device_nic}"
      PARENT_SCOPE)
  set(MORI_DEVICE_NIC_DEFINE
      "${_nic_define}"
      PARENT_SCOPE)
  set(MORI_IONIC_CCQE_DEFINE
      "${_ccqe_define}"
      PARENT_SCOPE)

  message(
    STATUS
      "Mori device config: arch=${_gpu_arch}, nic=${_device_nic}, define=${_nic_define} ${_ccqe_define}"
  )
endfunction()

# ---------------------------------------------------------------------------
# Public API: mori_add_device_target(<target>)
# ---------------------------------------------------------------------------
function(mori_add_device_target target)
  if(NOT DEFINED MORI_GPU_ARCH)
    message(
      FATAL_ERROR
        "Call mori_detect_device_config() before mori_add_device_target()")
  endif()

  if(MORI_DEVICE_NIC_DEFINE)
    target_compile_definitions(${target} PRIVATE ${MORI_DEVICE_NIC_DEFINE})
  endif()
  if(MORI_IONIC_CCQE_DEFINE)
    target_compile_definitions(${target} PRIVATE ${MORI_IONIC_CCQE_DEFINE})
  endif()
  target_compile_definitions(${target} PRIVATE HIP_ENABLE_WARP_SYNC_BUILTINS)

  if(MORI_GPU_ARCH)
    set_target_properties(${target} PROPERTIES HIP_ARCHITECTURES
                                               "${MORI_GPU_ARCH}")
  endif()
endfunction()
