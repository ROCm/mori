# ROCm 10's hsakmt-config.cmake runs find_dependency(NUMA) but ROCm ships no
# FindNUMA.cmake, so every consumer of find_package(hsakmt) has to supply one.
# Never invoked on ROCm <= 7.x, where that find_dependency call is commented
# out.
find_path(NUMA_INCLUDE_DIR NAMES numa.h)
find_library(NUMA_LIBRARY NAMES numa)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA REQUIRED_VARS NUMA_LIBRARY
                                                     NUMA_INCLUDE_DIR)

if(NUMA_FOUND AND NOT TARGET numa::numa)
  add_library(numa::numa UNKNOWN IMPORTED)
  set_target_properties(
    numa::numa PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
endif()

mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY)
